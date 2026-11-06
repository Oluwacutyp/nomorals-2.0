"""Assembly-audit truth-fix pass — AA-01 / AA-50 / AA-64 / AA-65 /
AA-67 / AA-68.

Each class drives the REAL path its finding named and asserts the
system now tells the truth (or the built thing is reachable):

- AA-01: importing the edit_engine package registers BOTH 3D rungs
  (rust + blender), and the resolver ladder walks them per profile.
- AA-50: /gear's conflict handler is reachable under its own name;
  the games gear handler keeps the plain name and its own context.
- AA-64: persona forget counts only records actually forgotten;
  an archive failure means NOT forgotten (integrity), named in the
  scheduled_pass report.
- AA-65: persona dedup returns ok: False + named failures when a
  merge fails; the clean-path shape is byte-identical to before.
- AA-67: both schedulers journal skipped/missed firings with a
  distinct status and ok=False — never as successful runs — and
  the run-history readers expose that state.
- AA-68: a failed chat-profile refresh returns the stored profile
  WITH stale markers (age + error), get_profile surfaces them to
  the context builder, and context lines say the profile is stale.

Teeth: every class fails on pristine c91e4c5 (the bootstrap import
state, the missing _control_conflict_gear, the missing detail
method, the ok=True skip records, the unmarked stale profile).
"""

from __future__ import annotations

import asyncio
import json
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_TREE_ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════════
# AA-01 — Blender rung registers in the package bootstrap
# ═══════════════════════════════════════════════════════════════════

class TestBlenderBootstrap:
    def test_fresh_interpreter_registers_both_rungs(self):
        code = (
            "import json\n"
            "from nomorals.media import edit_engine as ee\n"
            "from nomorals.media.edit_engine import (\n"
            "    registered_renderers, RENDERER_BOOTSTRAP_ISSUES)\n"
            "print(json.dumps({'rungs': registered_renderers(),\n"
            "                  'issues': RENDERER_BOOTSTRAP_ISSUES,\n"
            "                  'blender_cls': ee.BlenderRenderer.__name__,\n"
            "                  'rust_cls': ee.RustSceneRenderer.__name__}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=str(_TREE_ROOT),
            capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["issues"] == []
        assert out["rungs"] == ["rust3d", "blender"]
        assert out["blender_cls"] == "BlenderRenderer"
        assert out["rust_cls"] == "RustSceneRenderer"


_FAKE_BLENDER = """\
#!/usr/bin/env python3
import sys
if "--version" in sys.argv:
    print("Blender 4.2.3 (fake)")
    sys.exit(0)
sys.exit(2)
"""


def _fake_locator(tmp_path, available=True):
    from nomorals.media.edit_engine import render3d_blender as rb
    if available:
        fake = tmp_path / "fake-blender"
        fake.write_text(_FAKE_BLENDER, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXOTH)
        environ = {"NM_BLENDER_BIN": str(fake), "PATH": ""}
    else:
        environ = {"NM_BLENDER_BIN": "/nonexistent/blender",
                   "NM_BLENDER_NO_PROVISION": "1", "PATH": ""}
    return rb.BlenderLocator(tools_dir=tmp_path / "tools",
                             environ=environ)


def _ladder_scene():
    from nomorals.media.edit_engine.scene3d import (
        Camera,
        Material,
        Mesh,
        Rotation,
        Scene3D,
        SceneNode,
        Transform,
    )
    return Scene3D(
        nodes=[SceneNode(
            name="crate",
            mesh=Mesh(kind="primitive", primitive="cube",
                      params={"size": 2.0}),
            material="flatm")],
        cameras={"main": Camera(
            transform=Transform(position=(0, 0, 6),
                                rotation=Rotation.look_at((0, 0, 0))),
            fov=50.0)},
        active_camera="main",
        materials={"flatm": Material(kind="flat", color=(1.0, 0.0, 0.0))},
        tracks=[],
        output={"width": 64, "height": 64, "fps": 30},
    )


@pytest.fixture()
def _ladder(tmp_path):
    from nomorals.media.edit_engine import render3d_blender as rb
    from nomorals.media.edit_engine import render3d_rust as rr
    from nomorals.media.edit_engine.scene3d import reset_renderers
    reset_renderers()
    yield rb, rr
    reset_renderers()
    rr.register_rust_renderer()
    rb.register_blender_renderer()


class TestRendererLadderProfiles:
    def test_both_registered_rust_serves_first(self, _ladder, tmp_path):
        from nomorals.media.edit_engine.scene3d import registered_renderers, resolve_renderer
        rb, rr = _ladder
        rr.register_rust_renderer()
        rb.register_blender_renderer(
            rb.BlenderRenderer(locator=_fake_locator(tmp_path)))
        assert registered_renderers() == ["rust3d", "blender"]
        renderer, _record = resolve_renderer(_ladder_scene())
        assert renderer.name == "rust3d"
        # named selection reaches the Blender rung directly
        named, _rec = resolve_renderer(_ladder_scene(), name="blender")
        assert named.name == "blender"

    def test_blender_only_profile(self, _ladder, tmp_path):
        from nomorals.media.edit_engine.scene3d import resolve_renderer
        rb, _rr = _ladder
        rb.register_blender_renderer(
            rb.BlenderRenderer(locator=_fake_locator(tmp_path)))
        renderer, _record = resolve_renderer(_ladder_scene())
        assert renderer.name == "blender"

    def test_rust_only_profile(self, _ladder):
        from nomorals.media.edit_engine.scene3d import resolve_renderer
        _rb, rr = _ladder
        rr.register_rust_renderer()
        renderer, _record = resolve_renderer(_ladder_scene())
        assert renderer.name == "rust3d"

    def test_unavailable_blender_falls_through_to_rust(
            self, _ladder, tmp_path):
        from nomorals.media.edit_engine.scene3d import resolve_renderer
        rb, rr = _ladder
        rb.register_blender_renderer(rb.BlenderRenderer(
            locator=_fake_locator(tmp_path, available=False)))
        rr.register_rust_renderer()
        renderer, record = resolve_renderer(_ladder_scene())
        assert renderer.name == "rust3d"
        assert record["refused"][0]["renderer"] == "blender"

    def test_empty_registry_refuses(self, _ladder):
        from nomorals.media.edit_engine.scene3d import Scene3DError, resolve_renderer
        with pytest.raises(Scene3DError):
            resolve_renderer(_ladder_scene())


# ═══════════════════════════════════════════════════════════════════
# AA-50 — /gear dispatch: conflict vs arena handlers un-shadowed
# ═══════════════════════════════════════════════════════════════════

class TestGearDispatch:
    def test_mro_bindings_are_the_right_modules(self):
        from nomorals.agents.partner.runtime import PartnerRuntime
        assert (PartnerRuntime._control_gear.__module__
                == "nomorals.agents.partner.runtime_games")
        assert (PartnerRuntime._control_conflict_gear.__module__
                == "nomorals.agents.partner.runtime_group")

    def test_group_context_reaches_the_loadout_handler(self):
        from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

        captured = {}

        class _Store:
            @staticmethod
            def settings(key):
                return None

        class _Surface:
            store = _Store()

            def _conflict_gate(self, key, kind):
                return None

            def loadout_view(self, key, user_id):
                return {"power": 3, "rep": 2, "stars": 0,
                        "items": [], "defenses": {}}

        class _Self:
            def _economy_gate(self, message):
                return "telegram:-100", None

            def _group_surface(self):
                return _Surface()

            def _group_admin_ok(self, message):
                return False

            def _card(self, message, card, buttons=None):
                captured["card"] = card
                captured["buttons"] = buttons
                return "LOADOUT-CARD"

        message = SimpleNamespace(sender_id="42", sender="amy")
        out = RuntimeGroupMixin._control_conflict_gear(
            _Self(), "", message)
        assert out == "LOADOUT-CARD"
        assert "no player here" not in out
        assert captured["buttons"], "loadout card carries its buttons"

    def test_games_context_keeps_its_own_handler(self):
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

        stub = SimpleNamespace(context=SimpleNamespace(db=None))
        out = RuntimeGamesMixin._control_gear(
            stub, "inventory", "", player=None)
        # the arena handler answers in its own voice (feature gate or
        # the player gate) — it is NOT the group loadout path.
        assert ("no player here" in out) or ("games are off" in out)


# ═══════════════════════════════════════════════════════════════════
# AA-64 / AA-65 — persona curation tells the truth per record
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture()
def _memory(tmp_path):
    from nomorals.agents.context import build_context
    from nomorals.core.config import Settings
    ctx = build_context(Settings(home=str(tmp_path)))
    ctx.__enter__()
    yield ctx.memory
    ctx.__exit__(None, None, None)


def _doomed(memory, content):
    from nomorals.memory.base import MemoryKind
    rid = memory.remember(content, kind=MemoryKind.EPISODE,
                          importance=0.01, source="test")
    memory.db.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (time.time() - 90 * 86400, rid))
    return rid


class TestForgetTruth:
    def test_forget_failure_is_named_and_not_counted(
            self, _memory, monkeypatch):
        from nomorals.memory.persona import MemoryCurator
        good = _doomed(_memory, "cruft one")
        bad = _doomed(_memory, "cruft two")
        orig_forget = _memory.forget

        def flaky(rid):
            if rid == bad:
                raise RuntimeError("delete boom")
            return orig_forget(rid)

        monkeypatch.setattr(_memory, "forget", flaky)
        curator = MemoryCurator(_memory)
        detail = curator._forget_with_archive_detail(0.08)
        assert detail["forgotten"] == 1
        assert [f["id"] for f in detail["failed"]] == [bad]
        assert "delete boom" in detail["failed"][0]["error"]
        assert detail["archive_failed"] == []
        assert _memory.get(good) is None
        assert _memory.get(bad) is not None  # still there — truthfully
        # and the scheduled report carries the same truth
        report = curator.scheduled_pass()
        assert report["forgotten"] == 0  # bad one still can't go
        assert report["forget_failed"][0]["id"] == bad

    def test_archive_failure_means_not_forgotten(
            self, _memory, monkeypatch):
        from nomorals.memory.persona import MemoryCurator
        good = _doomed(_memory, "cruft one")
        bad = _doomed(_memory, "cruft two")
        curator = MemoryCurator(_memory)
        orig_archive = curator._archive

        def flaky_archive(rid, reason="", superseded_by=""):
            if rid == bad:
                return False
            return orig_archive(rid, reason=reason,
                                superseded_by=superseded_by)

        monkeypatch.setattr(curator, "_archive", flaky_archive)
        detail = curator._forget_with_archive_detail(0.08)
        assert detail["forgotten"] == 1
        assert detail["archive_failed"] == [bad]
        assert detail["failed"] == []
        # integrity: no archive copy → the record was NOT deleted
        assert _memory.get(bad) is not None
        assert _memory.get(good) is None
        rows = _memory.db.query(
            "SELECT id FROM memory_archive WHERE id = ?", (good,))
        assert rows, "the forgotten record's archive copy exists"


class TestDedupTruth:
    def _pair(self, memory):
        from nomorals.memory.base import MemoryKind
        a = memory.remember("I prefer short answers",
                            kind=MemoryKind.PREFERENCE, importance=0.9)
        b = memory.remember("I prefer short answers!",
                            kind=MemoryKind.PREFERENCE, importance=0.4)
        return a, b

    def test_merge_failure_reports_not_ok(self, _memory, monkeypatch):
        from nomorals.memory.base import MemoryKind
        from nomorals.memory.persona import MemoryCurator
        a, b = self._pair(_memory)

        def boom(rid):
            raise RuntimeError("forget boom")

        monkeypatch.setattr(_memory, "forget", boom)
        report = MemoryCurator(_memory).deduplicate(
            kind=MemoryKind.PREFERENCE)
        assert report["ok"] is False
        assert report["merged"] == 0
        assert [f["id"] for f in report["failures"]] == [b]
        assert _memory.get(b) is not None  # merge failed → both stay

    def test_clean_merge_keeps_the_exact_old_shape(self, _memory):
        from nomorals.memory.base import MemoryKind
        from nomorals.memory.persona import MemoryCurator
        a, b = self._pair(_memory)
        report = MemoryCurator(_memory).deduplicate(
            kind=MemoryKind.PREFERENCE)
        assert report["ok"] is True
        assert set(report) == {"ok", "merged", "kept"}
        assert report["merged"] >= 1
        assert _memory.get(b) is None


# ═══════════════════════════════════════════════════════════════════
# AA-67 — schedulers journal non-runs as non-runs
# ═══════════════════════════════════════════════════════════════════

def _core_scheduler():
    from nomorals.scheduler.scheduler import Scheduler
    from nomorals.storage.db import Database
    db = Database(":memory:")
    sched = Scheduler(db)
    calls = []

    async def ok_handler(**kwargs):
        calls.append(kwargs)

    sched.register_action("ok", ok_handler)
    return sched, db, calls


def _core_cron_row(db, task_id):
    return dict(db.query_one(
        """SELECT c.*, t.action, t.parameters, t.metadata, t.task_type,
                  t.status, t.created_at FROM cron_jobs c
           JOIN scheduled_tasks t ON c.task_id = t.task_id
           WHERE c.task_id = ?""", (task_id,)))


class TestCoreSchedulerRunTruth:
    def test_missed_cron_journals_missed_not_success(self):
        sched, db, _calls = _core_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        db.execute("UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
                   (time.time() - 90000, job.task_id))
        ran = asyncio.run(
            sched._fire_cron(_core_cron_row(db, job.task_id), time.time()))
        assert ran is False
        runs = sched.recent_runs(job.task_id)
        assert runs[0]["status"] == "missed"
        assert runs[0]["ok"] is False

    def test_policy_skip_journals_skipped(self):
        sched, db, _calls = _core_scheduler()
        job = asyncio.run(sched.schedule_cron(
            None, "* * * * *", "ok", missed_fire_policy="skip"))
        db.execute("UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
                   (time.time() - 120, job.task_id))
        ran = asyncio.run(
            sched._fire_cron(_core_cron_row(db, job.task_id), time.time()))
        assert ran is False
        runs = sched.recent_runs(job.task_id)
        assert runs[0]["status"] == "skipped"
        assert runs[0]["ok"] is False

    def test_overlap_skip_journals_skipped(self):
        sched, db, _calls = _core_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        sched._mark_flight(job.task_id, True)
        ran = asyncio.run(
            sched._fire_cron(_core_cron_row(db, job.task_id), time.time()))
        assert ran is False
        runs = sched.recent_runs(job.task_id)
        assert runs[0]["status"] == "skipped"
        assert runs[0]["ok"] is False

    def test_real_run_still_journals_ran_ok(self):
        sched, db, calls = _core_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        ran = asyncio.run(
            sched._fire_cron(_core_cron_row(db, job.task_id), time.time()))
        assert ran is True and len(calls) == 1
        runs = sched.recent_runs(job.task_id)
        assert runs[0]["status"] == "ran"
        assert runs[0]["ok"] is True

    def test_record_run_guard_never_journals_a_non_run_as_ok(self):
        sched, db, _calls = _core_scheduler()
        sched._record_run("t1", time.time(), 0.0, True,
                          "skipped: forced", "tick", status="skipped")
        row = db.query_one("SELECT ok, status FROM task_runs "
                           "WHERE task_id = 't1'")
        assert row["ok"] == 0 and row["status"] == "skipped"


def _agents_scheduler():
    from nomorals.agents.scheduler import Scheduler
    from nomorals.storage.db import Database

    class _Tools:
        def call(self, tool, capabilities=None, **args):
            return SimpleNamespace(ok=True, value={"ok": True},
                                   error=None)

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(db=db, tools=_Tools(),
                          settings=SimpleNamespace(), extras={})
    sched = Scheduler(ctx)
    sched.notifier = SimpleNamespace(
        publish=lambda *a, **k: {"delivered": True, "channel": "test"})
    return sched


def _agents_row(sched, job_id):
    # the raw schedule_jobs row — the shape the tick's own internals
    # consume (list_jobs humanizes fields like spec for display).
    return dict(sched.db.query_one(
        "SELECT * FROM schedule_jobs WHERE id = ?", (job_id,)))


class TestAgentsSchedulerRunTruth:
    def test_missed_fire_skip_journals_skipped(self):
        sched = _agents_scheduler()
        job = sched.add("missed skip job", "every 1h", "tool",
                        {"tool": "echo"}, missed_fire_policy="skip")
        with sched.db.transaction():
            sched.db.execute(
                "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                (time.time() - 7200, job["id"]))
        out = sched._apply_missed_fire(
            _agents_row(sched, job["id"]), time.time())
        assert out is None  # nothing executed
        runs = sched.recent_runs(job["id"])
        assert runs[0]["status"] == "skipped"
        assert runs[0]["ok"] is False

    def test_blackout_and_overlap_skips_journal_skipped(self):
        sched = _agents_scheduler()
        job = sched.add("skip job", "every 1h", "tool", {"tool": "echo"})
        row = _agents_row(sched, job["id"])
        sched._skip_blackout(row, time.time())
        sched._skip_overlap(row, time.time())
        runs = sched.recent_runs(job["id"])
        assert [r["status"] for r in runs] == ["skipped", "skipped"]
        assert all(r["ok"] is False for r in runs)

    def test_real_run_still_journals_ran_ok(self):
        sched = _agents_scheduler()
        job = sched.add("real job", "every 1h", "tool", {"tool": "echo"})
        out = sched.run_now(job["id"])
        assert out["ok"] is True
        runs = sched.recent_runs(job["id"])
        assert runs[0]["status"] == "ran"
        assert runs[0]["ok"] is True

    def test_record_run_guard(self):
        sched = _agents_scheduler()
        job = sched.add("guard job", "every 1h", "tool", {"tool": "echo"})
        sched._record_run(job["id"], time.time(), 0.0, True,
                          "skipped: forced", 0, "tick", status="skipped")
        runs = sched.recent_runs(job["id"])
        assert runs[0]["status"] == "skipped"
        assert runs[0]["ok"] is False


# ═══════════════════════════════════════════════════════════════════
# AA-68 — chat-profile refresh failure surfaces staleness
# ═══════════════════════════════════════════════════════════════════

class _ProfileDB:
    """chat_profiles + messages stand-in with fault flags."""

    def __init__(self, messages=()):
        self.row = None
        self.messages = list(messages)
        self.fail_aggregate = False
        self.upserts = []

    def execute(self, sql, params=()):
        if "INSERT INTO chat_profiles" in sql:
            self.upserts.append(params)
            self.row = {"profile_json": params[1],
                        "message_count": params[2],
                        "updated_at": params[3]}
        return None

    def query_one(self, sql, params=()):
        return self.row

    def query(self, sql, params=()):
        if "GROUP BY name" in sql:
            if self.fail_aggregate:
                raise RuntimeError("aggregate boom")
            counts = {}
            for m in self.messages:
                counts[m["name"]] = counts.get(m["name"], 0) + 1
            return [{"name": n, "n": c} for n, c in counts.items()]
        if "FROM messages" in sql:
            return [{"content": m["content"]} for m in self.messages]
        return []


def _seed_profile(db, updated_at):
    db.row = {
        "profile_json": json.dumps(
            {"participants": {"amy": 5}, "topics": ["gold"],
             "first_seen": time.time() - 86400}),
        "message_count": 5,
        "updated_at": updated_at,
    }


class TestChatProfileStaleness:
    def test_failed_refresh_marks_the_stored_profile_stale(self):
        from nomorals.partner import chat_profile as cp
        db = _ProfileDB(messages=[{"name": "amy", "content": "gold"}])
        _seed_profile(db, time.time() - 3600)
        db.fail_aggregate = True
        out = cp.refresh_profile(db, "chat-1")
        # the fallback copy itself carries the truth
        assert out["participants"] == {"amy": 5}
        assert out["stale"] is True
        assert "aggregate boom" in out["refresh_error"]
        assert out["profile_age_s"] >= 3500
        # and the consumer path (get_profile → context lines) sees it
        prof = cp.get_profile(db, "chat-1")
        assert prof["stale"] is True
        lines = cp.build_context_lines(prof)
        assert any("STALE" in line for line in lines)

    def test_successful_refresh_clears_the_stale_mark(self):
        from nomorals.partner import chat_profile as cp
        msgs = [{"name": "amy", "content": "gold gold rally"}] * 3
        db = _ProfileDB(messages=msgs)
        _seed_profile(db, time.time() - 3600)
        db.fail_aggregate = True
        assert cp.refresh_profile(db, "chat-2")["stale"] is True
        db.fail_aggregate = False
        out = cp.refresh_profile(db, "chat-2")
        assert "stale" not in out
        assert "stale" not in cp.get_profile(db, "chat-2")

    def test_fresh_profile_has_no_stale_mark(self):
        from nomorals.partner import chat_profile as cp
        msgs = [{"name": "amy", "content": "gold gold rally"}] * 3
        db = _ProfileDB(messages=msgs)
        out = cp.refresh_profile(db, "chat-3")
        assert "stale" not in out
        assert "stale" not in cp.get_profile(db, "chat-3")
        lines = cp.build_context_lines(cp.get_profile(db, "chat-3"))
        assert not any("STALE" in line for line in lines)
