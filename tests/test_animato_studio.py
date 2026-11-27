"""Tests for Animato Studio (port step 1): StudioProject + spark → build → share.

The render and publish steps run against fakes at the organ
boundaries — the pipeline and the publisher are injected. The
orchestration under test: project persistence, the spark flow
(plan → preflight → render → publish/export), honest failure
reporting, and the CLI + chat + tool wiring.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.media.contentops.studio import (  # noqa: E402
    CastMember,
    Episode,
    StudioProject,
    StudioRunner,
    StudioStore,
    slugify,
)

# ── fakes ────────────────────────────────────────────────────────────


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
                 render_ok: bool = True, render_error: str = ""):
        self.runs_root = root
        self.preflight_ok = preflight_ok
        self.render_ok = render_ok
        self.render_error = render_error
        self.planned: list[dict] = []
        self._n = 0

    def plan(self, niche, topic, *, style="", footage="", voice="",
             music="", cast=None):
        self._n += 1
        job = FakeJob(f"job_fake{self._n}", topic)
        self.planned.append({"niche": niche, "topic": topic, "style": style,
                             "footage": footage, "voice": voice,
                             "music": music, "cast": list(cast or [])})
        return job

    def preflight(self, job):
        if self.preflight_ok:
            return {"ok": True, "checks": [
                {"name": "ffmpeg", "ok": True, "detail": "/usr/bin/ffmpeg"}]}
        return {"ok": False, "checks": [
            {"name": "ffmpeg", "ok": False, "detail": "not on PATH"},
            {"name": "niche", "ok": True, "detail": "did_you_know"}]}

    def run(self, job):
        job.run_id = f"run_fake{self._n}"
        run_dir = Path(self.runs_root) / job.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / "final.mp4"
        if not self.render_ok:
            return FakeResult(False, job.run_id, job.id, None,
                              self.render_error or "render exploded")
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
            raise RuntimeError("YouTube credentials missing "
                               "(simulated — connect OAuth first)")
        self.published.append({"path": path, "title": title,
                               "privacy": privacy})
        return {"video_id": "yt_fake123",
                "url": "https://youtube.com/watch?v=yt_fake123"}


def _runner(tmp_path, **kw):
    return StudioRunner(tmp_path / "studio",
                        pipeline=kw.pop("pipeline", None),
                        publisher_factory=kw.pop("publisher_factory", None))


# ── project model ────────────────────────────────────────────────────


def test_slugify():
    assert slugify("My Cooking Show") == "my-cooking-show"
    assert slugify("  Stories & News!  ") == "stories-news"
    assert slugify("") == "studio"
    assert slugify("!!!") == "studio"


def test_project_round_trip_ignores_unknown_keys():
    project = StudioProject(
        slug="stories", name="Stories", niche="reddit_stories",
        cast=[CastMember(name="Ada", voice="edge", face_ref="ada.png")],
        topic_queue=["a lighthouse keeper"],
        episodes=[Episode(id="ep_1", topic="t", status="ready")])
    restored = StudioProject.from_dict(
        {**project.to_dict(), "future_field": "ignored"})
    assert restored.slug == "stories"
    assert restored.cast[0].name == "Ada"
    assert restored.cast[0].face_ref == "ada.png"
    assert restored.topic_queue == ["a lighthouse keeper"]
    assert restored.episodes[0].id == "ep_1"


def test_store_create_load_list_duplicate(tmp_path):
    store = StudioStore(tmp_path / "studio")
    created = store.create("My Cooking Show", niche="did_you_know",
                           music="lofi", schedule="daily")
    assert created.slug == "my-cooking-show"
    assert created.music == "lofi" and created.schedule == "daily"
    loaded = store.load("my-cooking-show")
    assert loaded.name == "My Cooking Show"
    assert [p.slug for p in store.list()] == ["my-cooking-show"]
    with pytest.raises(ValueError, match="already exists"):
        store.create("My Cooking Show")
    with pytest.raises(FileNotFoundError):
        store.load("nope")


def test_store_queue_topic_and_corrupt_manifest_skipped(tmp_path):
    store = StudioStore(tmp_path / "studio")
    store.create("Stories")
    store.queue_topic("stories", "a lighthouse keeper")
    assert store.load("stories").topic_queue == ["a lighthouse keeper"]
    bad = tmp_path / "studio" / "projects" / "broken" / "manifest.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json", encoding="utf-8")
    assert [p.slug for p in store.list()] == ["stories"]


def test_episode_lookup_by_prefix():
    project = StudioProject(slug="s", name="S",
                            episodes=[Episode(id="ep_abcdef", topic="t")])
    assert project.episode("ep_abc").id == "ep_abcdef"
    with pytest.raises(KeyError):
        project.episode("ep_zzz")


# ── spark flow ───────────────────────────────────────────────────────


def test_spark_happy_path_publishes_and_records_episode(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    publisher = FakePublisher()
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: publisher)
    project = runner.store.create("Stories", niche="motivation")
    result = runner.spark(project, "discipline beats motivation",
                          publish="youtube")
    assert result["ok"] is True
    assert result["video_path"].endswith("final.mp4")
    assert result["published"]["youtube"] == "yt_fake123"
    assert result["rungs"]["publish"] == "posted to youtube"
    saved = runner.store.load("stories")
    assert len(saved.episodes) == 1
    assert saved.episodes[0].status == "posted"
    assert saved.episodes[0].published["youtube"] == "yt_fake123"
    assert pipeline.planned[0]["niche"] == "motivation"
    assert pipeline.planned[0]["voice"] == "edge"
    assert publisher.published[0]["privacy"] == "public"


def test_spark_export_only_returns_the_file(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create("Stories")
    result = runner.spark(project, "why the moon looks big")
    assert result["ok"] is True
    assert result["published"] == {}
    assert result["rungs"]["publish"] == "skipped (export only)"
    assert runner.store.load("stories").episodes[0].status == "ready"


def test_spark_uses_queued_topic_and_consumes_it_on_success(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create("Stories")
    runner.store.queue_topic("stories", "a lighthouse keeper")
    result = runner.spark(project)
    assert result["ok"] is True
    saved = runner.store.load("stories")
    assert saved.topic_queue == []
    assert saved.episodes[0].topic == "a lighthouse keeper"


def test_spark_cast_voice_wins_over_project_voice(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create(
        "Stories", voice="edge",
        cast=[CastMember(name="Ada", voice="ada_warm")])
    runner.spark(project, "hello")
    assert pipeline.planned[0]["voice"] == "ada_warm"


def test_spark_no_idea_fails_closed(tmp_path):
    runner = _runner(tmp_path, pipeline=FakePipeline(tmp_path / "runs"))
    project = runner.store.create("Stories")
    result = runner.spark(project)
    assert result["ok"] is False
    assert "no idea" in result["rungs"]["spark"]
    assert runner.store.load("stories").episodes == []


def test_spark_unknown_project_fails_closed(tmp_path):
    runner = _runner(tmp_path, pipeline=FakePipeline(tmp_path / "runs"))
    result = runner.spark("ghost", "hello")
    assert result["ok"] is False
    assert "unknown project" in result["rungs"]["project"]


def test_spark_preflight_failure_keeps_topic_queued(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs", preflight_ok=False)
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create("Stories")
    runner.store.queue_topic("stories", "a lighthouse keeper")
    result = runner.spark(project)
    assert result["ok"] is False
    assert "failed: ffmpeg" in result["rungs"]["preflight"]
    assert "preflight failed" in result["notes"][0]
    saved = runner.store.load("stories")
    assert saved.topic_queue == ["a lighthouse keeper"]
    assert saved.episodes[0].status == "failed"


def test_spark_render_failure_records_episode(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs", render_ok=False,
                            render_error="ffmpeg died")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create("Stories")
    result = runner.spark(project, "hello")
    assert result["ok"] is False
    assert "ffmpeg died" in result["rungs"]["render"]
    saved = runner.store.load("stories")
    assert saved.episodes[0].status == "failed"
    assert "ffmpeg died" in saved.episodes[0].error


def test_spark_publish_failure_still_returns_the_video(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher(fail=True))
    project = runner.store.create("Stories")
    result = runner.spark(project, "hello", publish="youtube")
    assert result["ok"] is True  # the video exists; only the post failed
    assert "not posted" in result["rungs"]["publish"]
    assert "credentials missing" in result["notes"][-1]
    assert runner.store.load("stories").episodes[0].status == "ready"


def test_spark_publish_without_factory_fails_closed(tmp_path):
    """No publisher anywhere → the video is kept, the rung names the wire.

    The L4 studio never imports the L5 publisher itself, so a bare
    runner with no factory (and no composition-root default) must
    fail the publish rung honestly, not raise or silently skip.
    """
    from nomorals.media.contentops.studio import set_publisher_factory

    set_publisher_factory(None)
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create("Stories")
    result = runner.spark(project, "hello", publish="youtube")
    assert result["ok"] is True  # the video exists; only the post failed
    assert "not posted" in result["rungs"]["publish"]
    assert "no YouTube publisher configured" in result["notes"][-1]
    assert runner.store.load("stories").episodes[0].status == "ready"


def test_set_publisher_factory_wires_bare_runner(tmp_path):
    """The composition-root slot is what the tool/autopost paths use."""
    from nomorals.media.contentops.studio import set_publisher_factory

    publisher = FakePublisher()
    set_publisher_factory(lambda: publisher)
    try:
        pipeline = FakePipeline(tmp_path / "runs")
        runner = _runner(tmp_path, pipeline=pipeline)
        project = runner.store.create("Stories")
        result = runner.spark(project, "hello", publish="youtube")
        assert result["ok"] is True
        assert result["published"]["youtube"] == "yt_fake123"
    finally:
        set_publisher_factory(None)


def test_publish_and_export_episode(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    publisher = FakePublisher()
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: publisher)
    project = runner.store.create("Stories")
    result = runner.spark(project, "hello")
    episode_id = result["episode_id"]
    exported = runner.export_episode("stories", episode_id)
    assert exported["ok"] is True
    assert exported["path"] == result["video_path"]
    posted = runner.publish_episode("stories", episode_id)
    assert posted["ok"] is True
    assert posted["video_id"] == "yt_fake123"
    assert runner.store.load("stories").episodes[0].status == "posted"
    with pytest.raises(KeyError):
        runner.export_episode("stories", "ep_nope")


def test_publish_episode_wrong_status_and_missing_file(tmp_path):
    runner = _runner(tmp_path, pipeline=FakePipeline(tmp_path / "runs"))
    project = runner.store.create("Stories")
    project.episodes.append(Episode(id="ep_bad", topic="t",
                                    status="failed"))
    runner.store.save(project)
    result = runner.publish_episode("stories", "ep_bad")
    assert result["ok"] is False
    assert "not ready" in result["reason"]
    project.episodes.append(Episode(id="ep_ghost", topic="t",
                                    status="ready",
                                    video_path="/no/such/final.mp4"))
    runner.store.save(project)
    result = runner.export_episode("stories", "ep_ghost")
    assert result["ok"] is False
    assert "no rendered video" in result["reason"]


# ── CLI ──────────────────────────────────────────────────────────────


def _cli_args(action, **kw):
    args = SimpleNamespace(animato_action=action, json=False,
                           project="", name="", idea="", topic="",
                           episode="", niche="did_you_know", style="",
                           voice="edge", music="", channel="",
                           schedule="", publish="", private=False)
    args.__dict__.update(kw)
    return args


def test_cli_new_list_show_queue_status(tmp_path, monkeypatch, capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("new", name="Stories",
                                 niche="motivation"), None) == 0
    assert "stories" in capsys.readouterr().out
    assert cmd_animato(_cli_args("list"), None) == 0
    assert "stories" in capsys.readouterr().out
    assert cmd_animato(_cli_args("show", name="stories"), None) == 0
    out = capsys.readouterr().out
    assert "motivation" in out
    assert cmd_animato(_cli_args("queue", name="stories",
                                 topic="a lighthouse keeper"), None) == 0
    assert cmd_animato(_cli_args("status", project="stories"), None) == 0
    assert "no episodes yet" in capsys.readouterr().out
    assert cmd_animato(_cli_args("new", name="Stories"), None) == 1
    assert "already exists" in capsys.readouterr().err


def test_cli_unknown_action_and_missing_project(tmp_path, monkeypatch):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("dance"), None) == 1
    assert cmd_animato(_cli_args("show", name="ghost"), None) == 1


def test_parser_accepts_animato_spark():
    from nomorals.cmdline import _parser

    parser = _parser()
    args = parser.parse_args(
        ["animato", "spark", "why the moon looks big",
         "--publish", "youtube"])
    assert args.animato_action == "spark"
    assert args.idea == "why the moon looks big"
    assert args.publish == "youtube"
    args = parser.parse_args(["animato", "new", "Stories",
                              "--niche", "motivation"])
    assert args.animato_action == "new"
    assert args.niche == "motivation"


def test_dispatch_registers_animato_once():
    from nomorals.cmdline import _COMMANDS, _parser

    assert "animato" in _COMMANDS
    assert _COMMANDS["animato"].handler is not None
    parser = _parser()
    assert "animato" in parser._subparsers._group_actions[0].choices


# ── chat + tool wiring ───────────────────────────────────────────────


def test_chat_registry_covers_animato():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        list_catalog,
    )

    assert "animato" in CONTROL_COMMANDS
    assert "animato" in COMMAND_DETAILS
    assert COMMAND_DETAILS["animato"]["what"]
    assert COMMAND_DETAILS["animato"]["usage"]
    assert COMMAND_DETAILS["animato"]["example"]
    assert "animato" in LIST_ONELINERS
    slots = [kind for _group, kinds in LIST_GROUPS for kind in kinds
             if kind == "animato"]
    assert slots == ["animato"]
    assert "/animato" in list_catalog()


def test_tool_registers_animato_spark():
    from nomorals.tools.animato import animato_spark, register

    seen: dict = {}

    class FakeRegistry:
        def register(self, name, fn, **kw):
            seen[name] = (fn, kw)

    register(FakeRegistry())
    assert "animato_spark" in seen
    fn, kw = seen["animato_spark"]
    assert fn is animato_spark
    assert kw["capability"] == "media"
    assert kw["examples"]


# ── port step 4: character-consistency hardening ─────────────────────


def _write_png(path: Path, color: tuple = (200, 30, 40)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_resolve_cast_lock_ok_skips_voice_only_members(tmp_path):
    from nomorals.media.contentops.studio import resolve_cast_lock

    ref = _write_png(tmp_path / "ada.png")
    lock, error = resolve_cast_lock([
        CastMember(name="Ada", voice="ada_warm", face_ref=str(ref)),
        CastMember(name="Ben", voice="edge"),  # no reference: no lock
    ])
    assert error == ""
    assert [m["name"] for m in lock] == ["Ada"]
    assert lock[0]["face_ref"] == str(ref)
    assert lock[0]["voice"] == "ada_warm"
    lock2, error2 = resolve_cast_lock([])
    assert (lock2, error2) == ([], "")


def test_resolve_cast_lock_missing_reference_fails_closed(tmp_path):
    from nomorals.media.contentops.studio import resolve_cast_lock

    lock, error = resolve_cast_lock([
        CastMember(name="Ada", face_ref=str(tmp_path / "ghost.png"))])
    assert lock == []
    assert "Ada" in error and "not found" in error


def test_resolve_cast_lock_unreadable_reference_fails_closed(tmp_path):
    from nomorals.media.contentops.studio import resolve_cast_lock

    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"this is not an image at all")
    lock, error = resolve_cast_lock([CastMember(name="Ada", face_ref=str(bad))])
    assert lock == []
    assert "Ada" in error and "not a readable image" in error
    directory = tmp_path / "refs"
    directory.mkdir()
    lock, error = resolve_cast_lock([
        CastMember(name="Ada", face_ref=str(directory))])
    assert lock == []
    assert "not a file" in error


def test_store_add_cast_adds_and_replaces_by_name(tmp_path):
    from nomorals.media.contentops.studio import resolve_cast_lock

    store = StudioStore(tmp_path / "studio")
    store.create("Stories")
    ref = _write_png(tmp_path / "ada.png")
    project = store.add_cast(
        "stories", CastMember(name="Ada", face_ref=str(ref)))
    assert project.cast[0].name == "Ada"
    assert Path(project.cast[0].face_ref).is_absolute()
    # same name replaces (a re-shot reference), it never duplicates
    ref2 = _write_png(tmp_path / "ada2.png", (10, 200, 30))
    project = store.add_cast(
        "stories", CastMember(name="Ada", voice="ada_warm",
                              face_ref=str(ref2)))
    assert len(project.cast) == 1
    assert project.cast[0].voice == "ada_warm"
    assert project.cast[0].face_ref == str(ref2)
    assert store.load("stories").cast[0].face_ref == str(ref2)
    lock, error = resolve_cast_lock(store.load("stories").cast)
    assert error == "" and lock[0]["face_ref"] == str(ref2)


def test_spark_passes_cast_lock_into_every_episode(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    ref = _write_png(tmp_path / "ada.png")
    ref2 = _write_png(tmp_path / "ben.png", (10, 200, 30))
    project = runner.store.create(
        "Stories", cast=[CastMember(name="Ada", face_ref=str(ref)),
                         CastMember(name="Ben", face_ref=str(ref2))])
    first = runner.spark(project, "Ada opens the lighthouse door")
    second = runner.spark(project, "Ben climbs the lighthouse stairs")
    assert first["ok"] is True and second["ok"] is True
    # the SAME lock reaches the render for every episode of the series
    assert len(pipeline.planned) == 2
    for planned in pipeline.planned:
        assert [m["name"] for m in planned["cast"]] == ["Ada", "Ben"]
        assert planned["cast"][0]["face_ref"] == str(ref)
        assert planned["cast"][1]["face_ref"] == str(ref2)
    assert first["rungs"]["cast"] == "locked: Ada, Ben"
    assert second["rungs"]["cast"] == "locked: Ada, Ben"


def test_spark_missing_reference_fails_closed_before_planning(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create(
        "Stories", cast=[CastMember(
            name="Ada", face_ref=str(tmp_path / "ghost.png"))])
    result = runner.spark(project, "Ada opens the lighthouse door")
    assert result["ok"] is False
    assert "Ada" in result["rungs"]["cast"]
    assert "not found" in result["rungs"]["cast"]
    assert pipeline.planned == []  # nothing planned, no render spend
    saved = runner.store.load("stories")
    assert saved.episodes[0].status == "failed"
    assert "not found" in saved.episodes[0].error
    assert saved.episodes[0].video_path == ""


def test_spark_voice_only_cast_renders_exactly_as_before(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline)
    project = runner.store.create(
        "Stories", cast=[CastMember(name="Ada", voice="ada_warm")])
    result = runner.spark(project, "hello")
    assert result["ok"] is True
    assert pipeline.planned[0]["cast"] == []
    assert "cast" not in result["rungs"]


def test_cli_cast_adds_member_and_rejects_missing_file(
        tmp_path, monkeypatch, capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("new", name="Stories"), None) == 0
    ref = _write_png(tmp_path / "ada.png")
    assert cmd_animato(_cli_args(
        "cast", name="stories", member="Ada", voice="ada_warm",
        face_ref=str(ref), character_id=""), None) == 0
    out = capsys.readouterr().out
    assert "Ada" in out and "stories" in out
    assert cmd_animato(_cli_args(
        "cast", name="stories", member="Ghost",
        face_ref=str(tmp_path / "ghost.png")), None) == 1
    assert "no face reference image" in capsys.readouterr().err
    assert cmd_animato(_cli_args("show", name="stories"), None) == 0
    assert "Ada" in capsys.readouterr().out


def test_parser_accepts_animato_cast():
    from nomorals.cmdline import _parser

    parser = _parser()
    args = parser.parse_args(
        ["animato", "cast", "stories", "Ada",
         "--voice", "ada_warm", "--face-ref", "/tmp/ada.png"])
    assert args.animato_action == "cast"
    assert args.name == "stories"
    assert args.member == "Ada"
    assert args.face_ref == "/tmp/ada.png"


# ── port step 5: GitHub-studio export ────────────────────────────────


def _export_project(tmp_path, *, topics=("a lighthouse keeper",
                                        "the tide that never came"),
                    cast=True):
    """A runner + a project with queued topics (and a cast, by default)."""
    runner = _runner(tmp_path)
    runner.store.create("Stories", niche="motivation", style="warm",
                        voice="edge", music="lofi", channel="stories",
                        schedule="daily")
    for topic in topics:
        runner.store.queue_topic("stories", topic)
    if cast:
        ref = _write_png(tmp_path / "ada.png")
        runner.store.add_cast(
            "stories", CastMember(name="Ada", voice="ada_warm",
                                  face_ref=str(ref)))
    return runner, runner.store.load("stories")


def test_export_project_writes_brief_manifest_and_scripts(tmp_path):
    runner, project = _export_project(tmp_path)
    dest = tmp_path / "out"
    result = runner.export_project("stories", dest=dest)
    assert result["ok"] is True
    assert result["project"] == "stories"
    assert result["path"] == str(dest)
    assert result["topics"] == 2
    assert result["cast"] == ["Ada"]
    for expected in ("BRIEF.md", "README.md", "manifest.json",
                     "render.py", "render.sh", "cast/ada.png"):
        assert expected in result["files"]
    stored = json.loads((tmp_path / "studio" / "projects" / "stories"
                         / "manifest.json").read_text(encoding="utf-8"))
    exported = json.loads((dest / "manifest.json").read_text(
        encoding="utf-8"))
    assert exported == stored
    brief = (dest / "BRIEF.md").read_text(encoding="utf-8")
    assert "motivation" in brief and "daily" in brief
    assert "a lighthouse keeper" in brief
    assert "the tide that never came" in brief
    assert "Ada" in brief
    compile((dest / "render.py").read_text(encoding="utf-8"),
            "render.py", "exec")
    render_sh = (dest / "render.sh").read_text(encoding="utf-8")
    assert render_sh.startswith("#!/bin/sh")
    assert "render.py" in render_sh
    assert (dest / "cast" / "ada.png").read_bytes() == (
        tmp_path / "ada.png").read_bytes()


def test_export_project_scripts_reference_real_topics_and_cast(tmp_path):
    runner, project = _export_project(tmp_path)
    dest = tmp_path / "out"
    assert runner.export_project("stories", dest=dest)["ok"] is True
    script = (dest / "render.py").read_text(encoding="utf-8")
    assert "a lighthouse keeper" in script
    assert "the tide that never came" in script
    assert "Ada" in script
    assert "motivation" in script
    for filler in ("TODO", "FIXME", "YOUR TOPIC HERE", "placeholder"):
        assert filler not in script


def test_export_project_missing_project_fails_closed(tmp_path):
    runner = _runner(tmp_path)
    dest = tmp_path / "out"
    result = runner.export_project("ghost", dest=dest)
    assert result["ok"] is False
    assert "ghost" in result["reason"]
    assert not dest.exists()


def test_export_project_no_topics_fails_closed(tmp_path):
    runner = _runner(tmp_path)
    runner.store.create("Stories")
    dest = tmp_path / "out"
    result = runner.export_project("stories", dest=dest)
    assert result["ok"] is False
    assert "no topics" in result["reason"]
    assert not dest.exists()


def test_export_project_unreadable_reference_fails_closed(tmp_path):
    runner = _runner(tmp_path)
    runner.store.create("Stories")
    runner.store.queue_topic("stories", "a lighthouse keeper")
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"this is not an image at all")
    runner.store.add_cast(
        "stories", CastMember(name="Ada", face_ref=str(bad)))
    dest = tmp_path / "out"
    result = runner.export_project("stories", dest=dest)
    assert result["ok"] is False
    assert "Ada" in result["reason"]
    assert "not a readable image" in result["reason"]
    assert not dest.exists()


def test_export_project_refuses_nonempty_destination(tmp_path):
    runner, project = _export_project(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "keep.txt").write_text("mine", encoding="utf-8")
    result = runner.export_project("stories", dest=dest)
    assert result["ok"] is False
    assert "not empty" in result["reason"]
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_export_project_without_cast(tmp_path):
    runner, project = _export_project(tmp_path, cast=False)
    dest = tmp_path / "out"
    result = runner.export_project("stories", dest=dest)
    assert result["ok"] is True
    assert result["cast"] == []
    brief = (dest / "BRIEF.md").read_text(encoding="utf-8")
    assert "## Cast" not in brief
    assert not (dest / "cast").exists()
    compile((dest / "render.py").read_text(encoding="utf-8"),
            "render.py", "exec")


def test_cli_export_project(tmp_path, monkeypatch, capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("new", name="Stories",
                                 niche="motivation"), None) == 0
    capsys.readouterr()
    assert cmd_animato(_cli_args("queue", name="stories",
                                 topic="a lighthouse keeper"), None) == 0
    dest = tmp_path / "out"
    assert cmd_animato(_cli_args("export", name="stories",
                                 dest=str(dest)), None) == 0
    out = capsys.readouterr().out
    assert "stories" in out and str(dest) in out
    assert (dest / "manifest.json").is_file()
    assert (dest / "render.py").is_file()
    # the old episode form still fails cleanly on an unknown episode
    assert cmd_animato(_cli_args("export", name="stories",
                                 episode="ep_nope"), None) == 1
    assert "ep_nope" in capsys.readouterr().err
    # and a missing project fails closed
    assert cmd_animato(_cli_args("export", name="ghost",
                                 dest=str(tmp_path / "out2")), None) == 1


def test_parser_accepts_animato_export():
    from nomorals.cmdline import _parser

    parser = _parser()
    args = parser.parse_args(
        ["animato", "export", "stories", "--dest", "/tmp/out"])
    assert args.animato_action == "export"
    assert args.name == "stories"
    assert args.episode == ""
    assert args.dest == "/tmp/out"
    args = parser.parse_args(["animato", "export", "stories", "ep_abc"])
    assert args.episode == "ep_abc"
    assert args.dest == ""
