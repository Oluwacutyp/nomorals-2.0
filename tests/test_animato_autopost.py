"""Tests for the Animato auto-post mission (port step 2).

The render and publish steps run against fakes at the organ
boundaries — the pipeline, publisher and calendar are injected.
The orchestration under test: schedule parsing, due-ness, the sweep
(queued topic → calendar topic → skip), honest failure journaling,
start/stop/status, the durable job, and the CLI + chat + tool wiring.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.media.contentops.autopost import (  # noqa: E402
    AUTOPOST_JOB,
    AutopostStore,
    autopost_status,
    ensure_animato_autopost_job,
    is_due,
    parse_schedule,
    run_autopost,
)
from nomorals.media.contentops.studio import (  # noqa: E402
    StudioRunner,
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


class FakeCalendarPost:
    def __init__(self, post_id: str, niche: str, topic: str):
        self.id = post_id
        self.niche = niche
        self.topic = topic
        self.status = "queued"

    def due_at(self):
        return datetime.now(timezone.utc) - timedelta(minutes=1)  # noqa: UP017


class FakeCalendar:
    """ContentCalendar-shaped fake: due posts + marks."""

    def __init__(self, posts=None):
        self.posts = list(posts or [])
        self.marks: list[tuple] = []

    def due(self, now=None):
        return [p for p in self.posts if p.status == "queued"]

    def mark(self, post_id, status, job_id=""):
        for p in self.posts:
            if p.id == post_id:
                p.status = status
        self.marks.append((post_id, status, job_id))


class FakePipeline:
    """ShortPipeline-shaped fake: plan → preflight → run, in-memory."""

    def __init__(self, root: Path, *, render_ok: bool = True,
                 render_error: str = "", calendar=None):
        self.runs_root = root
        self.render_ok = render_ok
        self.render_error = render_error
        self.calendar = calendar if calendar is not None else FakeCalendar()
        self.planned: list[dict] = []
        self._n = 0

    def plan(self, niche, topic, *, style="", footage="", voice=""):
        self._n += 1
        job = FakeJob(f"job_fake{self._n}", topic)
        self.planned.append({"niche": niche, "topic": topic, "style": style,
                             "footage": footage, "voice": voice})
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


def _store(tmp_path):
    return AutopostStore(tmp_path / "studio")


def _now():
    return datetime.now(timezone.utc)  # noqa: UP017


# ── schedule parsing ─────────────────────────────────────────────────


def test_parse_schedule_shapes():
    assert parse_schedule("daily")["kind"] == "daily"
    assert parse_schedule("Weekly")["kind"] == "weekly"
    assert parse_schedule("  daily  ")["kind"] == "daily"
    assert parse_schedule("")["kind"] == "none"
    assert parse_schedule("   ")["kind"] == "none"
    once = parse_schedule("2026-12-01T09:00:00+00:00")
    assert once["kind"] == "once"
    assert once["at"] == datetime(2026, 12, 1, 9, 0, tzinfo=timezone.utc)  # noqa: UP017
    naive = parse_schedule("2026-12-01T09:00:00")
    assert naive["at"].tzinfo is timezone.utc  # noqa: UP017
    with pytest.raises(ValueError, match="unparseable schedule"):
        parse_schedule("every tuesday")
    with pytest.raises(ValueError, match="unparseable schedule"):
        parse_schedule("soon")


def test_is_due_matrix():
    now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)  # noqa: UP017
    daily = parse_schedule("daily")
    assert is_due(daily, "", now) is True
    assert is_due(daily, "2026-11-27T01:00:00+00:00", now) is False
    assert is_due(daily, "2026-11-26T23:00:00+00:00", now) is True
    weekly = parse_schedule("weekly")
    assert is_due(weekly, "", now) is True
    assert is_due(weekly, "2026-11-24T12:00:00+00:00", now) is False
    assert is_due(weekly, "2026-11-20T12:00:00+00:00", now) is True
    once = parse_schedule("2026-11-27T09:00:00+00:00")
    assert is_due(once, "", now) is True
    assert is_due(once, "2026-11-27T10:00:00+00:00", now) is False
    assert is_due(once, "2026-11-27T08:00:00+00:00", now) is True
    future = parse_schedule("2026-11-28T09:00:00+00:00")
    assert is_due(future, "", now) is False
    assert is_due(parse_schedule(""), "", now) is False


# ── mission state ────────────────────────────────────────────────────


def test_autopost_store_round_trip_and_cap(tmp_path):
    store = _store(tmp_path)
    assert store.enabled is False
    assert store.last_attempt("stories") == ""
    assert store.set_enabled(True) is True
    assert AutopostStore(tmp_path / "studio").enabled is True
    for i in range(205):
        store.record_run({"project": "stories", "topic": f"t{i}",
                          "episode_id": "", "status": "posted",
                          "video_path": "", "video_id": "yt1",
                          "reason": ""})
    assert len(store.runs) == 200
    assert store.runs[0]["topic"] == "t5"
    assert store.last_attempt("stories")
    assert store.projects["stories"]["last_status"] == "posted"
    assert [r["topic"] for r in store.recent(2)] == ["t204", "t203"]


def test_autopost_store_corrupt_file_starts_empty(tmp_path):
    root = tmp_path / "studio"
    root.mkdir(parents=True, exist_ok=True)
    (root / "autopost.json").write_text("{not json", encoding="utf-8")
    store = AutopostStore(root)
    assert store.enabled is False
    assert store.runs == []


# ── the sweep ────────────────────────────────────────────────────────


def test_run_posts_due_project_and_journals_it(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    publisher = FakePublisher()
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: publisher)
    runner.store.create("Stories", niche="motivation", schedule="daily")
    runner.store.queue_topic("stories", "discipline beats motivation")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["ok"] is True
    assert result["ran"] == 1
    assert result["posted"] == 1
    assert result["failed"] == 0
    outcome = result["outcomes"][0]
    assert outcome["status"] == "posted"
    assert outcome["video_id"] == "yt_fake123"
    assert outcome["topic"] == "discipline beats motivation"
    saved = runner.store.load("stories")
    assert saved.topic_queue == []
    assert saved.episodes[0].status == "posted"
    assert publisher.published[0]["privacy"] == "public"
    assert store.last_attempt("stories")
    assert store.recent(1)[0]["status"] == "posted"
    # One attempt per slot: a second sweep today does nothing.
    again = run_autopost(runner=runner, store=store, now=_now())
    assert again["ran"] == 1
    assert again["outcomes"][0]["status"] == "not-due"
    assert len(pipeline.planned) == 1


def test_run_while_stopped_does_nothing_unless_forced(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", schedule="daily")
    runner.store.queue_topic("stories", "hello")
    store = _store(tmp_path)
    assert store.enabled is False
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["ran"] == 0
    assert result["enabled"] is False
    assert pipeline.planned == []
    forced = run_autopost(runner=runner, store=store, now=_now(),
                          force=True)
    assert forced["ran"] == 1
    assert forced["posted"] == 1
    assert len(pipeline.planned) == 1


def test_run_skips_when_no_topic_queued(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", schedule="daily")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["skipped"] == 1
    outcome = result["outcomes"][0]
    assert outcome["status"] == "skipped"
    assert "no topic queued" in outcome["reason"]
    assert pipeline.planned == []
    assert store.recent(1)[0]["status"] == "skipped"


def test_run_takes_topic_from_due_calendar_post(tmp_path):
    calendar = FakeCalendar([FakeCalendarPost("post_1", "motivation",
                                             "morning pages")])
    pipeline = FakePipeline(tmp_path / "runs", calendar=calendar)
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", niche="motivation", schedule="daily")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["posted"] == 1
    assert result["outcomes"][0]["topic"] == "morning pages"
    assert calendar.posts[0].status == "posted"
    assert ("post_1", "rendering", "") in calendar.marks
    assert any(m[0] == "post_1" and m[1] == "posted"
               for m in calendar.marks)


def test_run_publish_failure_journals_reason_and_keeps_video(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher(fail=True))
    runner.store.create("Stories", schedule="daily")
    runner.store.queue_topic("stories", "hello")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["failed"] == 1
    outcome = result["outcomes"][0]
    assert outcome["status"] == "failed"
    assert "credentials missing" in outcome["reason"]
    assert outcome["video_path"].endswith("final.mp4")
    assert outcome["video_id"] == ""
    saved = runner.store.load("stories")
    assert saved.episodes[0].status == "ready"
    assert store.recent(1)[0]["reason"] == outcome["reason"]


def test_run_render_failure_journals_reason(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs", render_ok=False,
                            render_error="ffmpeg died")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", schedule="daily")
    runner.store.queue_topic("stories", "hello")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["failed"] == 1
    assert "ffmpeg died" in result["outcomes"][0]["reason"]
    assert result["outcomes"][0]["video_path"] == ""
    assert runner.store.load("stories").episodes[0].status == "failed"


def test_run_unparseable_schedule_skips_with_reason(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", schedule="every tuesday")
    runner.store.queue_topic("stories", "hello")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now())
    assert result["skipped"] == 1
    assert "unparseable schedule" in result["outcomes"][0]["reason"]
    assert pipeline.planned == []


def test_run_unknown_project_fails_closed(tmp_path):
    runner = _runner(tmp_path, pipeline=FakePipeline(tmp_path / "runs"))
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now(),
                          project="ghost")
    assert result["ok"] is False
    assert "unknown studio project" in result["reason"]


def test_run_project_filter_and_limit(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", schedule="daily")
    runner.store.queue_topic("stories", "one")
    runner.store.create("Cooking", schedule="daily")
    runner.store.queue_topic("cooking", "two")
    store = _store(tmp_path)
    store.set_enabled(True)
    result = run_autopost(runner=runner, store=store, now=_now(),
                          project="cooking")
    assert result["ran"] == 1
    assert result["outcomes"][0]["project"] == "cooking"
    assert len(pipeline.planned) == 1
    limited = run_autopost(runner=runner, store=store, now=_now(),
                           limit=0, force=True)
    assert limited["ran"] == 0


def test_status_reports_switch_slots_and_runs(tmp_path):
    pipeline = FakePipeline(tmp_path / "runs")
    runner = _runner(tmp_path, pipeline=pipeline,
                     publisher_factory=lambda: FakePublisher())
    runner.store.create("Stories", schedule="daily")
    runner.store.queue_topic("stories", "hello")
    store = _store(tmp_path)
    status = autopost_status(runner=runner, store=store, now=_now())
    assert status["enabled"] is False
    assert status["projects"][0]["due"] is True
    assert status["projects"][0]["queued_topics"] == 1
    store.set_enabled(True)
    run_autopost(runner=runner, store=store, now=_now())
    status = autopost_status(runner=runner, store=store, now=_now())
    assert status["projects"][0]["due"] is False
    assert status["projects"][0]["last_status"] == "posted"
    assert len(status["recent"]) == 1


# ── durable job ──────────────────────────────────────────────────────


class FakeScheduler:
    def __init__(self):
        self.jobs: list[dict] = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, **kw):
        job = {"id": f"job_{len(self.jobs)}", "name": name, "spec": spec,
               "kind": kind, "payload": payload, **kw}
        self.jobs.append(job)
        return job


def test_ensure_autopost_job_idempotent():
    sched = FakeScheduler()
    first = ensure_animato_autopost_job(sched)
    assert first[0]["scheduled"] is True
    assert first[0]["name"] == AUTOPOST_JOB
    job = sched.jobs[0]
    assert job["spec"] == "every 30m"
    assert job["payload"] == {"tool": "animato_autopost",
                              "args": {"action": "run"}}
    assert job["routine"] is True
    second = ensure_animato_autopost_job(sched)
    assert second[0]["already_scheduled"] is True
    assert len(sched.jobs) == 1


# ── CLI ──────────────────────────────────────────────────────────────


def _cli_args(action, *, animato_action="autopost", **kw):
    args = SimpleNamespace(animato_action=animato_action,
                           autopost_action=action, json=False,
                           project="", name="", idea="", topic="",
                           episode="", niche="did_you_know", style="",
                           voice="edge", music="", channel="",
                           schedule="", publish="", private=False,
                           force=False)
    args.__dict__.update(kw)
    return args


def test_cli_autopost_start_stop_status_run(tmp_path, monkeypatch, capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("status"), None) == 0
    assert "autopost: off" in capsys.readouterr().out
    assert cmd_animato(_cli_args("start"), None) == 0
    assert "started" in capsys.readouterr().out
    assert cmd_animato(_cli_args("status"), None) == 0
    assert "autopost: on" in capsys.readouterr().out
    assert cmd_animato(_cli_args("stop"), None) == 0
    assert "stopped" in capsys.readouterr().out
    # A stopped mission's run does nothing; --force runs it.
    assert cmd_animato(_cli_args("new", animato_action="new",
                                 name="Stories", schedule="daily"),
                       None) == 0
    assert cmd_animato(_cli_args("run"), None) == 0
    assert "autopost is off" in capsys.readouterr().out
    assert cmd_animato(_cli_args("run", force=True), None) == 0
    out = capsys.readouterr().out
    assert "ran 1 project(s)" in out
    assert "1 skipped" in out  # no topic queued


def test_cli_autopost_status_json_and_unknown_project(tmp_path, monkeypatch,
                                                     capsys):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.cmdline.commands.animato import cmd_animato

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert cmd_animato(_cli_args("status", json=True), None) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is False
    assert cmd_animato(_cli_args("run", project="ghost"), None) == 1
    assert "unknown studio project" in capsys.readouterr().err


def test_parser_accepts_animato_autopost():
    from nomorals.cmdline import _parser

    parser = _parser()
    args = parser.parse_args(["animato", "autopost", "run",
                              "--project", "stories", "--force"])
    assert args.animato_action == "autopost"
    assert args.autopost_action == "run"
    assert args.project == "stories"
    assert args.force is True
    args = parser.parse_args(["animato", "autopost", "status"])
    assert args.autopost_action == "status"


# ── chat + tool wiring ───────────────────────────────────────────────


def test_tool_registers_animato_autopost():
    from nomorals.tools.animato import animato_autopost, register

    seen: dict = {}

    class FakeRegistry:
        def register(self, name, fn, **kw):
            seen[name] = (fn, kw)

    register(FakeRegistry())
    assert "animato_autopost" in seen
    assert "animato_spark" in seen
    fn, kw = seen["animato_autopost"]
    assert fn is animato_autopost
    assert kw["capability"] == "media"
    assert kw["examples"]


def test_tool_autopost_start_stop_status(tmp_path, monkeypatch):
    import nomorals.media.contentops.studio as studio_mod
    from nomorals.tools.animato import animato_autopost

    monkeypatch.setattr(studio_mod, "DEFAULT_STUDIO_ROOT",
                        tmp_path / "studio")
    assert animato_autopost("status")["enabled"] is False
    assert animato_autopost("start")["enabled"] is True
    assert animato_autopost("stop")["enabled"] is False
    assert animato_autopost("dance")["ok"] is False


def test_chat_registry_usage_mentions_autopost():
    from nomorals.social.chat.control import COMMAND_DETAILS

    assert "animato" in COMMAND_DETAILS
    assert "autopost" in COMMAND_DETAILS["animato"]["usage"]
