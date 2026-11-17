"""Gapfix: the mission render path must consult the profile gate.

``resolve_render_profile`` (media/edit_engine/render_queue.py) clamps
termux/mobile/embedded renders to 480p with ``degraded=True`` and an
explicit reason, but its only production caller was ``LocalHost.plan``
— the mission path rendered the ``full`` stage through the raw stored
spec with no profile read, no clamp, no degraded flag. These tests pin
the gate on the mission path, written BEFORE the wiring (failing-first).
"""

from __future__ import annotations

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


def _enqueue(store, tmp_path, **kw):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"data")
    return enqueue_render(store, sources=[src], spec=_spec(), **kw)


def _fake_pure_factory(seen, tmp_path):
    """Fake render_pure: records the spec it was asked to render and
    returns a deterministic output path per height."""

    def fake_render_pure(sources, spec, pack, *, cache_dir, renderer=None):
        seen.append(dict(spec))
        height = int(spec.get("height") or 0)
        out = tmp_path / f"render-{height}.mp4"
        out.write_bytes(b"fake-video")
        return {"output": str(out), "cache_key": "k", "cached": False}

    return fake_render_pure


class TestFullStageProfileGate:
    def test_full_stage_clamps_on_termux(self, store, ctx, tmp_path,
                                         monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import run_render_tick_once

        seen: list[dict] = []
        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory(seen, tmp_path))
        mission = _enqueue(store, tmp_path)
        run_render_tick_once(ctx, store=store, profile="termux")
        run_render_tick_once(ctx, store=store, profile="termux")
        status = render_mission_status(store, mission.id)
        assert status["done"] is True
        # proxy (480p by design) then the full stage, clamped to 480p.
        assert [int(s["height"]) for s in seen] == [480, 480]
        # width scales to preserve aspect: 1080 * 480 / 1920 = 270.
        assert [int(s["width"]) for s in seen] == [270, 270]
        assert status["degraded"] is True
        assert status["full_height"] == 480
        assert status["profile"] == "termux"
        assert "termux" in status["degraded_reason"]
        assert "allow_full_res" in status["degraded_reason"]

    def test_full_stage_full_res_on_workstation(self, store, ctx, tmp_path,
                                                monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import run_render_tick_once

        seen: list[dict] = []
        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory(seen, tmp_path))
        mission = _enqueue(store, tmp_path)
        run_render_tick_once(ctx, store=store, profile="workstation")
        run_render_tick_once(ctx, store=store, profile="workstation")
        status = render_mission_status(store, mission.id)
        assert status["done"] is True
        assert [int(s["height"]) for s in seen] == [480, 1920]
        assert status["degraded"] is False
        assert status["degraded_reason"] == ""
        assert status["full_height"] == 1920
        assert status["profile"] == "workstation"

    def test_small_spec_on_termux_is_not_degraded(self, store, ctx,
                                                  tmp_path, monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import run_render_tick_once

        seen: list[dict] = []
        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory(seen, tmp_path))
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")
        mission = enqueue_render(store, sources=[src],
                                 spec=_spec(height=480, width=270))
        run_render_tick_once(ctx, store=store, profile="termux")
        run_render_tick_once(ctx, store=store, profile="termux")
        status = render_mission_status(store, mission.id)
        assert status["done"] is True
        assert [int(s["height"]) for s in seen] == [480, 480]
        assert status["degraded"] is False
        assert status["degraded_reason"] == ""

    def test_proxy_stage_is_not_a_downgrade(self, store, ctx, tmp_path,
                                            monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import run_render_tick_once

        seen: list[dict] = []
        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory(seen, tmp_path))
        mission = _enqueue(store, tmp_path)
        run_render_tick_once(ctx, store=store, profile="mobile")
        m = store.get(mission.id)
        proxy = m.state["render"]["stages"]["proxy"]
        assert proxy["degraded"] is False
        assert proxy["reason"] == ""
        assert proxy["height"] == 480

    def test_tick_profile_defaults_to_detected(self, store, ctx, tmp_path,
                                               monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import run_render_tick_once

        seen: list[dict] = []
        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory(seen, tmp_path))
        monkeypatch.setattr(
            driver, "detect_profile",
            lambda: SimpleNamespace(kind="embedded"))
        mission = _enqueue(store, tmp_path)
        run_render_tick_once(ctx, store=store)
        run_render_tick_once(ctx, store=store)
        status = render_mission_status(store, mission.id)
        assert status["degraded"] is True
        assert status["profile"] == "embedded"
        assert "embedded" in status["degraded_reason"]

    def test_stage_render_fn_direct(self, tmp_path, monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import stage_render_fn

        seen: list[dict] = []
        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory(seen, tmp_path))
        state = {"spec": _spec(), "sources": ["a.mp4"], "pack": {}}
        result = stage_render_fn("full", state, cache_dir=tmp_path,
                                 profile="termux")
        assert result["degraded"] is True
        assert result["height"] == 480
        assert "termux" in result["reason"]
        result = stage_render_fn("full", state, cache_dir=tmp_path,
                                 profile="pc")
        assert result["degraded"] is False
        assert result["height"] == 1920
        assert result["reason"] == ""

    def test_degraded_visible_in_status_text(self, store, ctx, tmp_path,
                                             monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.cmdline.commands.render import _status_text
        from nomorals.missions.render_driver import run_render_tick_once

        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory([], tmp_path))
        mission = _enqueue(store, tmp_path)
        run_render_tick_once(ctx, store=store, profile="termux")
        run_render_tick_once(ctx, store=store, profile="termux")
        text = _status_text(render_mission_status(store, mission.id))
        assert "degraded" in text
        assert "480p" in text
        assert "termux" in text

    def test_degraded_visible_in_done_notification(self, store, ctx,
                                                   tmp_path, monkeypatch):
        import nomorals.missions.render_driver as driver
        from nomorals.missions.render_driver import run_render_tick_once

        class _FakeNotifier:
            def __init__(self):
                self.published: list[tuple[str, str, str]] = []

            def publish(self, kind, title="", body="", **_kw):
                self.published.append((kind, title, body))
                return {"ok": True}

        monkeypatch.setattr(driver, "render_pure",
                            _fake_pure_factory([], tmp_path))
        notifier = _FakeNotifier()
        _enqueue(store, tmp_path)
        run_render_tick_once(ctx, store=store, profile="termux",
                             notifier=notifier)
        run_render_tick_once(ctx, store=store, profile="termux",
                             notifier=notifier)
        assert len(notifier.published) == 1
        kind, _title, body = notifier.published[0]
        assert kind == "render"
        assert "degraded" in body
        assert "480p" in body
        assert "termux" in body

    def test_same_triple_second_mission_hits_cache_on_termux(
            self, store, ctx, tmp_path):
        """The gate is a pure function of (profile, height): the
        clamped full spec is byte-identical to the proxy spec, so on a
        small profile the full stage is a cache hit — one real render
        per triple, never per mission."""
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
            from nomorals.missions.render_driver import stage_render_fn

            return stage_render_fn(stage, state, cache_dir=cache_dir,
                                   profile="termux",
                                   renderer=counting_renderer)

        m1 = _enqueue(store, tmp_path)
        m2 = _enqueue(store, tmp_path)
        assert m1.id != m2.id
        for _ in range(2):
            run_render_tick_once(ctx, store=store, render_fn=stage_fn)
        assert render_mission_status(store, m1.id)["done"] is True
        assert render_mission_status(store, m2.id)["done"] is True
        assert render_mission_status(store, m1.id)["degraded"] is True
        assert render_mission_status(store, m2.id)["degraded"] is True
        # One proxy render for the shared triple; both clamped full
        # stages and the second mission's stages were cache hits.
        assert len(renderer_calls) == 1
        assert (render_mission_status(store, m1.id)["outputs"]
                == render_mission_status(store, m2.id)["outputs"])

    def test_same_triple_workstation_still_renders_full_once(
            self, store, ctx, tmp_path):
        """Contrast: on a workstation the full stage is a genuinely
        different render from the proxy, so the triple costs two."""
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
            from nomorals.missions.render_driver import stage_render_fn

            return stage_render_fn(stage, state, cache_dir=cache_dir,
                                   profile="workstation",
                                   renderer=counting_renderer)

        m1 = _enqueue(store, tmp_path)
        m2 = _enqueue(store, tmp_path)
        for _ in range(2):
            run_render_tick_once(ctx, store=store, render_fn=stage_fn)
        assert render_mission_status(store, m1.id)["done"] is True
        assert render_mission_status(store, m2.id)["done"] is True
        assert render_mission_status(store, m1.id)["degraded"] is False
        assert len(renderer_calls) == 2
