"""Gapfix R2-33: the mission render queue must actually run.

``nomorals/missions/render_queue.py`` was complete and test-pinned but
had zero production callers: no tick ever advanced a render mission,
nothing could enqueue one, and a failed stage was recorded nowhere.
These tests pin the driver (``missions/render_driver.py``) + the
``nm render`` CLI verb, written BEFORE the wiring (failing-first).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.missions import MissionStore
from nomorals.missions.render_queue import (
    enqueue_render,
    render_mission_status,
)
from nomorals.storage.db import Database


def _spec(**over):
    base = {"clips": [{"path": "a.mp4", "start": 0.0, "end": 2.0}],
            "segments": [[0, 2.0]], "fps": 30, "width": 1080,
            "height": 1920, "seed": 0}
    base.update(over)
    return base


@pytest.fixture()
def store():
    db = Database(":memory:")
    db.migrate()
    return MissionStore(db)


@pytest.fixture()
def ctx(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = Database(str(tmp_path / "test.db"))
    db.migrate()
    return SimpleNamespace(settings=SimpleNamespace(home=str(home)), db=db)


class _FakeNotifier:
    def __init__(self):
        self.published: list[tuple[str, str, str]] = []

    def publish(self, kind, title="", body="", **_kw):
        self.published.append((kind, title, body))
        return {"ok": True}


class _FakeScheduler:
    def __init__(self):
        self.jobs: list[dict] = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"id": f"job-{len(self.jobs)}", "name": name, "spec": spec,
               "kind": kind, "payload": payload, "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


def _enqueue(store, tmp_path, **kw):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"data")
    return enqueue_render(store, sources=[src], spec=_spec(), **kw)


class TestRenderTick:
    def test_tick_advances_one_stage_per_tick_and_reaches_done(
            self, store, ctx, tmp_path):
        from nomorals.missions.render_driver import run_render_tick_once

        calls: list[str] = []

        def render_fn(*, stage, state):
            calls.append(stage)
            return {"output": f"/tmp/{stage}.mp4"}

        mission = _enqueue(store, tmp_path)
        r1 = run_render_tick_once(ctx, store=store, render_fn=render_fn)
        assert r1["ok"] is True
        assert r1["advanced"] == [mission.id]
        assert r1["done"] == []
        assert calls == ["proxy"]
        status = render_mission_status(store, mission.id)
        assert status["completed_stages"] == ["proxy"]
        assert status["done"] is False
        r2 = run_render_tick_once(ctx, store=store, render_fn=render_fn)
        assert r2["done"] == [mission.id]
        assert calls == ["proxy", "full"]
        assert render_mission_status(store, mission.id)["done"] is True

    def test_crash_mid_stage_resumes_without_rerunning_completed(
            self, store, ctx, tmp_path):
        from nomorals.missions.render_driver import run_render_tick_once

        calls: list[str] = []

        def render_fn(*, stage, state):
            calls.append(stage)
            if stage == "proxy" and calls.count("proxy") == 1:
                raise RuntimeError("simulated crash mid-proxy")
            return {"output": f"/tmp/{stage}.mp4"}

        mission = _enqueue(store, tmp_path)
        r1 = run_render_tick_once(ctx, store=store, render_fn=render_fn)
        assert r1["advanced"] == []
        assert r1["failed"] == []
        # The failure is recorded on the mission, not dropped, and the
        # mission is NOT failed yet — one crash is a retry, not a verdict.
        m = store.get(mission.id)
        render = m.state["render"]
        assert render["consecutive_failures"] == 1
        assert render["failed_stage"] == "proxy"
        assert "simulated crash" in render["last_error"]
        assert render_mission_status(store, mission.id)[
            "completed_stages"] == []
        assert m.status != "failed"
        run_render_tick_once(ctx, store=store, render_fn=render_fn)
        run_render_tick_once(ctx, store=store, render_fn=render_fn)
        assert calls == ["proxy", "proxy", "full"]
        assert render_mission_status(store, mission.id)["done"] is True
        # A later tick never re-runs a completed stage.
        run_render_tick_once(ctx, store=store, render_fn=render_fn)
        assert calls == ["proxy", "proxy", "full"]

    def test_status_reports_progress_and_budget_spent(
            self, store, ctx, tmp_path):
        from nomorals.missions.render_driver import run_render_tick_once

        mission = _enqueue(store, tmp_path, budget_wall=3600.0,
                           budget_tokens=1000)
        run_render_tick_once(
            ctx, store=store,
            render_fn=lambda *, stage, state: {"output": "/tmp/x.mp4"})
        status = render_mission_status(store, mission.id)
        assert status["completed_stages"] == ["proxy"]
        assert status["spent_wall"] > 0.0
        assert status["budget_wall"] == 3600.0
        assert status["budget_tokens"] == 1000
        assert status["spent_tokens"] == 0
        assert status["wall_remaining"] == pytest.approx(
            3600.0 - status["spent_wall"])

    def test_repeated_stage_failure_fails_mission_and_records_it(
            self, store, ctx, tmp_path):
        from nomorals.missions.render_driver import run_render_tick_once

        calls: list[str] = []

        def render_fn(*, stage, state):
            calls.append(stage)
            raise RuntimeError("ffmpeg blew up")

        mission = _enqueue(store, tmp_path)
        for _ in range(3):
            result = run_render_tick_once(
                ctx, store=store, render_fn=render_fn)
        assert result["failed"] == [mission.id]
        assert calls == ["proxy", "proxy", "proxy"]
        m = store.get(mission.id)
        assert m.status == "failed"
        assert "proxy" in (m.state.get("status_note") or "")
        assert "ffmpeg blew up" in m.state["render"]["last_error"]
        # A fourth tick never touches a terminal mission again.
        run_render_tick_once(ctx, store=store, render_fn=render_fn)
        assert calls == ["proxy", "proxy", "proxy"]

    def test_budget_exhausted_mission_never_renders(
            self, store, ctx, tmp_path):
        from nomorals.missions.render_driver import run_render_tick_once

        calls: list[str] = []
        mission = _enqueue(store, tmp_path, budget_wall=0.001)
        m = store.get(mission.id)
        m.charge(wall=1.0)
        store.save(m)
        result = run_render_tick_once(
            ctx, store=store,
            render_fn=lambda *, stage, state: (
                calls.append(stage), {"output": "/tmp/x.mp4"})[1])
        assert calls == []
        assert result["failed"] == [mission.id]
        assert store.get(mission.id).status == "failed"

    def test_done_notifies_once(self, store, ctx, tmp_path):
        from nomorals.missions.render_driver import run_render_tick_once

        notifier = _FakeNotifier()
        _enqueue(store, tmp_path)
        run_render_tick_once(
            ctx, store=store, notifier=notifier,
            render_fn=lambda *, stage, state: {"output": "/tmp/full.mp4"})
        run_render_tick_once(
            ctx, store=store, notifier=notifier,
            render_fn=lambda *, stage, state: {"output": "/tmp/full.mp4"})
        assert len(notifier.published) == 1
        kind, title, body = notifier.published[0]
        assert kind == "render"
        assert "/tmp/full.mp4" in body
        # A later tick never re-publishes a finished render.
        run_render_tick_once(
            ctx, store=store, notifier=notifier,
            render_fn=lambda *, stage, state: {"output": "/tmp/full.mp4"})
        assert len(notifier.published) == 1

    def test_same_triple_second_mission_hits_cache(
            self, store, ctx, tmp_path):
        """One queue owns the work: the second identical render is two
        cache hits through the shared engine cache — the renderer runs
        once per stage per triple, never once per mission."""
        from nomorals.media.edit_engine.pure_render import render_pure
        from nomorals.missions.render_driver import (
            default_render_cache_dir,
            run_render_tick_once,
        )

        renderer_calls: list = []
        cache_dir = default_render_cache_dir(ctx.settings.home)

        def counting_renderer(spec):
            renderer_calls.append(spec)
            out = Path(spec.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-video")
            return str(out)

        def stage_fn(*, stage, state):
            from nomorals.media.edit_engine.pure_render import proxy_spec

            eff = (proxy_spec(state["spec"]) if stage == "proxy"
                   else state["spec"])
            result = render_pure(state["sources"], eff, state["pack"],
                                 cache_dir=cache_dir,
                                 renderer=counting_renderer)
            return {"output": result["output"]}

        m1 = _enqueue(store, tmp_path)
        m2 = _enqueue(store, tmp_path)
        assert m1.id != m2.id
        for _ in range(2):
            run_render_tick_once(ctx, store=store, render_fn=stage_fn)
        assert render_mission_status(store, m1.id)["done"] is True
        assert render_mission_status(store, m2.id)["done"] is True
        # One proxy + one full render for the shared triple; the second
        # mission's stages were cache hits, not re-renders.
        assert len(renderer_calls) == 2
        assert (render_mission_status(store, m1.id)["outputs"]
                == render_mission_status(store, m2.id)["outputs"])

    def test_tick_never_raises_on_broken_store(self, ctx):
        from nomorals.missions.render_driver import run_render_tick_once

        broken = SimpleNamespace(settings=SimpleNamespace(home="/nonexistent"),
                                 db=None)
        result = run_render_tick_once(broken)
        assert result["ok"] is False
        assert "error" in result


class TestEnsureRenderTickJob:
    def test_job_is_idempotent(self):
        from nomorals.missions.render_driver import (
            RENDER_JOB,
            RENDER_SPEC,
            ensure_render_tick_job,
        )

        sched = _FakeScheduler()
        first = ensure_render_tick_job(sched)
        second = ensure_render_tick_job(sched)
        assert first is second
        assert len(sched.jobs) == 1
        assert sched.jobs[0]["name"] == RENDER_JOB
        assert sched.jobs[0]["spec"] == RENDER_SPEC
        stale = ensure_render_tick_job(sched, spec="every 1h")
        assert stale is not first
        assert len(sched.jobs) == 1
        assert sched.jobs[0]["spec"] == "every 1h"

    def test_none_scheduler_returns_none(self):
        from nomorals.missions.render_driver import ensure_render_tick_job

        assert ensure_render_tick_job(None) is None


class TestNmRenderCli:
    def _args(self, **kw):
        base = {"source": None, "spec": None, "pack": None, "name": "",
                "budget_wall": 0.0, "budget_tokens": 0, "status": "",
                "list": False, "advance": "", "json": False}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_enqueue_then_status_then_advance(self, ctx, tmp_path,
                                              monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.cmdline.commands.render import _cmd_render

        store = MissionStore(ctx.db)

        def fake_render_pure(sources, spec, pack, *, cache_dir, renderer=None):
            out = Path(cache_dir) / "fake.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-video")
            return {"output": str(out), "cache_key": "k", "cached": False}

        monkeypatch.setattr(driver, "render_pure", fake_render_pure)
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")
        rc = _cmd_render(self._args(source=[str(src)],
                                    spec=json.dumps(_spec())),
                         ctx)
        assert rc == 0
        missions = store.list(tag="render", active_only=True)
        assert len(missions) == 1
        mission_id = missions[0].id
        rc = _cmd_render(self._args(status=mission_id), ctx)
        assert rc == 0
        rc = _cmd_render(self._args(advance=mission_id), ctx)
        assert rc == 0
        rc = _cmd_render(self._args(advance=mission_id), ctx)
        assert rc == 0
        assert render_mission_status(store, mission_id)["done"] is True

    def test_list_shows_queued_renders(self, ctx, tmp_path):
        from nomorals.cmdline.commands.render import _cmd_render

        store = MissionStore(ctx.db)
        _enqueue(store, tmp_path)
        rc = _cmd_render(self._args(list=True), ctx)
        assert rc == 0

    def test_no_flags_is_usage_not_error(self, ctx):
        from nomorals.cmdline.commands.render import _cmd_render

        assert _cmd_render(self._args(), ctx) == 0
