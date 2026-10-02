"""Prompt 05 — project rooms: persistent per-goal workspaces.

Covers: room creation/layout, slug uniqueness, ROOM.md rehydration from a
fresh manager, adversarial path sandboxing (.., absolute, symlinks),
secret redaction on writes, decisions/blockers, links, search (no file
leak by default), crash/dirty reconciliation, tick bounds + blocker skip,
stale detection, goal/project step wrapping, auto-room creation, and
backward compatibility for roomless goals/projects.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.storage.db import Database
from nomorals.workspace.rooms import (
    RoomManager, RoomContext, RoomEscapeError, slugify,
)


def make_ctx(test=None):
    tmp = tempfile.mkdtemp(prefix="rooms-test-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db_path = os.path.join(tmp, "test.db")
    db = Database(db_path)
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp, rooms_auto_create=True)
    ctx = SimpleNamespace(db=db, settings=settings)
    return ctx, tmp


class SlugTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Q3 Tax Prep!"), "q3-tax-prep")

    def test_empty_falls_back(self):
        self.assertTrue(slugify("!!!").startswith("room-"))

    def test_max_60(self):
        self.assertLessEqual(len(slugify("a" * 200)), 60)


class CreateTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx(self)
        self.mgr = RoomManager(self.tmp, db=self.ctx.db)

    def test_layout(self):
        r = self.mgr.create("My Project", kind="project", linked_id="p1",
                            plan=["step one", "step two"])
        base = Path(self.tmp) / "rooms" / r.slug
        for sub in ("files", "scratch", "logs", "inbox"):
            self.assertTrue((base / sub).is_dir(), sub)
        self.assertTrue((base / "ROOM.md").is_file())
        self.assertTrue((base / "plan.md").is_file())
        self.assertIn("step one", (base / "plan.md").read_text())
        self.assertEqual(r.kind, "project")
        self.assertEqual(r.linked_id, "p1")

    def test_slug_uniqueness(self):
        a = self.mgr.create("Same Title")
        b = self.mgr.create("Same Title")
        self.assertNotEqual(a.slug, b.slug)
        self.assertTrue(b.slug.startswith(a.slug))

    def test_get_and_list(self):
        r = self.mgr.create("Listed")
        self.assertEqual(self.mgr.get(r.slug).id, r.id)
        self.assertIn(r.slug, [x.slug for x in self.mgr.list()])
        self.assertIn(r.slug,
                      [x.slug for x in self.mgr.list(status="active")])
        self.assertEqual(self.mgr.list(status="archived"), [])

    def test_get_by_linked(self):
        r = self.mgr.create("Linked Goal", kind="goal", linked_id="g123")
        found = self.mgr.get_by_linked("goal", "g123")
        self.assertIsNotNone(found)
        self.assertEqual(found.slug, r.slug)
        self.assertIsNone(self.mgr.get_by_linked("goal", "nope"))

    def test_lifecycle(self):
        r = self.mgr.create("Cycle")
        self.assertEqual(self.mgr.pause(r.slug).status, "paused")
        self.assertEqual(self.mgr.resume(r.slug).status, "active")
        self.assertEqual(self.mgr.archive(r.slug).status, "archived")
        self.assertEqual(self.mgr.list(status="archived")[0].slug, r.slug)


class RehydrationTests(unittest.TestCase):
    def test_fresh_manager_resumes_from_room_md(self):
        ctx, tmp = make_ctx(self)
        mgr = RoomManager(tmp, db=ctx.db)
        r = mgr.create("Resume Me", kind="goal", linked_id="g9")
        with mgr.enter(r.slug) as c:
            c.set_step("write the report")
            c.add_blocker("waiting on owner for figures")
            c.decide("use markdown", "portable")
        # fresh manager, same root+db — must resume from ROOM.md + state
        mgr2 = RoomManager(tmp, db=Database(os.path.join(tmp, "test.db")))
        mgr2.db.migrate()
        with mgr2.enter(r.slug) as c2:
            self.assertEqual(c2.room.current_step, "write the report")
            self.assertIn("waiting on owner for figures",
                          c2.room.blockers)
            self.assertEqual(len(c2.room.decisions), 1)

    def test_rehydrate_from_disk_when_db_row_missing(self):
        ctx, tmp = make_ctx(self)
        mgr = RoomManager(tmp, db=ctx.db)
        r = mgr.create("Orphan")
        with mgr.enter(r.slug) as c:
            c.set_step("halfway")
        # delete the DB row — the file alone must rebuild it
        ctx.db.execute("DELETE FROM rooms WHERE slug=?", (r.slug,))
        mgr2 = RoomManager(tmp, db=ctx.db)
        with mgr2.enter(r.slug) as c2:
            self.assertEqual(c2.room.current_step, "halfway")
            self.assertEqual(c2.room.slug, r.slug)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx(self)
        self.mgr = RoomManager(self.tmp, db=self.ctx.db)
        self.room = self.mgr.create("Sandbox")
        self.c = self.mgr.enter(self.room.slug)

    def tearDown(self):
        self.c.close()

    def test_dotdot_rejected(self):
        with self.assertRaises(RoomEscapeError):
            self.c.path("..", "evil.txt")

    def test_nested_dotdot_rejected(self):
        with self.assertRaises(RoomEscapeError):
            self.c.path("files", "..", "..", "etc", "passwd")

    def test_absolute_rejected(self):
        with self.assertRaises(RoomEscapeError):
            self.c.path("/etc/passwd")

    def test_symlink_out_rejected(self):
        # symlink inside the room pointing outside
        link = Path(self.tmp) / "rooms" / self.room.slug / "sneaky"
        link.symlink_to("/etc")
        with self.assertRaises(RoomEscapeError):
            self.c.path("sneaky", "passwd")

    def test_symlink_to_file_out_rejected(self):
        outside = Path(self.tmp) / "outside.txt"
        outside.write_text("secret")
        link = Path(self.tmp) / "rooms" / self.room.slug / "leak"
        link.symlink_to(outside)
        with self.assertRaises(RoomEscapeError):
            self.c.path("leak")

    def test_legit_paths_ok(self):
        p = self.c.path("files", "report.md")
        self.assertTrue(str(p).startswith(
            str(Path(self.tmp) / "rooms" / self.room.slug)))
        # symlink inside pointing inside is fine
        target = self.c.path("files", "real.txt")
        target.write_text("hi")
        link = Path(self.tmp) / "rooms" / self.room.slug / "ok-link"
        link.symlink_to(target)
        self.assertEqual(self.c.path("ok-link").read_text(), "hi")

    def test_dot_parts_normalized(self):
        p = self.c.path("files", ".", "x.md")
        self.assertTrue(str(p).endswith("files/x.md"))


class RedactionTests(unittest.TestCase):
    def test_secrets_redacted_in_logs_and_room_md(self):
        ctx, tmp = make_ctx(self)
        mgr = RoomManager(tmp, db=ctx.db)
        r = mgr.create("Redact")
        secret = "api_key=sk-live-1234567890abcdef"
        with mgr.enter(r.slug) as c:
            c.log("s1", "fetched", {"note": secret})
            c.decide("rotate key", secret)
        md = (Path(tmp) / "rooms" / r.slug / "ROOM.md").read_text()
        self.assertNotIn("sk-live", md)
        log = (Path(tmp) / "rooms" / r.slug / "logs" / "activity.log"
               ).read_text()
        self.assertNotIn("sk-live", log)


class LinkSearchTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx(self)
        self.mgr = RoomManager(self.tmp, db=self.ctx.db)

    def test_link(self):
        a = self.mgr.create("Alpha")
        b = self.mgr.create("Beta")
        self.mgr.link(a.slug, b.slug)
        md = (Path(self.tmp) / "rooms" / a.slug / "ROOM.md").read_text()
        self.assertIn(b.slug, md)

    def test_link_self_rejected(self):
        a = self.mgr.create("Solo")
        with self.assertRaises(ValueError):
            self.mgr.link(a.slug, a.slug)

    def test_search_finds_decisions_not_files(self):
        r = self.mgr.create("Searchable")
        with self.mgr.enter(r.slug) as c:
            c.decide("migrate to postgres", "scale")
            f = c.path("files", "notes.md")
            f.write_text("the launch code is banana")
        hits = self.mgr.search("postgres")
        self.assertTrue(any(h["slug"] == r.slug for h in hits))
        # file contents are NOT searched by default
        hits = self.mgr.search("banana")
        self.assertFalse(any(h["slug"] == r.slug for h in hits))
        # ... but --deep finds them
        hits = self.mgr.search("banana", deep=True)
        self.assertTrue(any(h["slug"] == r.slug and h["where"] != "room"
                            for h in hits))


class DirtyTests(unittest.TestCase):
    def test_dirty_reconciled_not_resumed(self):
        ctx, tmp = make_ctx(self)
        mgr = RoomManager(tmp, db=ctx.db)
        r = mgr.create("Crashy")
        ctx2 = mgr.enter(r.slug)
        ctx2.log("step-9", "start", {"description": "dangerous"})
        # simulate a crash: never checkpoint, never close — dirty stays set
        # (enter() set dirty=True; no checkpoint cleared it)
        mgr2 = RoomManager(tmp, db=ctx.db)
        with mgr2.enter(r.slug) as c:
            # reconciled: blocker added, NOT silently resumed
            self.assertTrue(any("unclean exit" in b for b in c.room.blockers))
            self.assertIn("dirty_tail", c.room.state)

    def test_clean_exit_clears_dirty(self):
        ctx, tmp = make_ctx(self)
        mgr = RoomManager(tmp, db=ctx.db)
        r = mgr.create("Clean")
        with mgr.enter(r.slug):
            pass
        # exiting the with-block checkpoints → dirty cleared
        row = ctx.db.query_one("SELECT state_json FROM rooms WHERE slug=?",
                               (r.slug,))
        self.assertFalse(json.loads(row["state_json"]).get("dirty"))


class TickTests(unittest.TestCase):
    def test_tick_advances_bounded_and_skips_blocked(self):
        ctx, tmp = make_ctx(self)
        from nomorals.agents.goals import GoalSystem
        gs = GoalSystem(ctx)
        # disable auto-room for the fixture goal; we'll link manually
        ctx.settings.rooms_auto_create = False
        goal = gs.create("Tick Goal", plan=["a", "b", "c", "d", "e"])
        mgr = RoomManager(tmp, db=ctx.db)
        room = mgr.create("Tick Room", kind="goal", linked_id=goal.id)
        calls = []
        out = mgr.tick(executor=lambda desc: calls.append(desc) or "ok",
                       goal_system=gs, max_steps_per_tick=2)
        self.assertEqual(len(calls), 2)  # bounded
        self.assertEqual(out["advanced"][0]["steps"], 2)
        # now block the room — tick must skip, not retry
        with mgr.enter(room.slug) as c:
            c.add_blocker("need owner approval")
            c.checkpoint("blocked")
        # clear the blocker flag from dirty-exit: checkpoint cleared dirty
        calls.clear()
        out = mgr.tick(executor=lambda desc: calls.append(desc) or "ok",
                       goal_system=gs, max_steps_per_tick=2)
        self.assertEqual(calls, [])
        self.assertEqual(len(out["skipped"]), 1)

    def test_stale_detection(self):
        ctx, tmp = make_ctx(self)
        mgr = RoomManager(tmp, db=ctx.db)
        r = mgr.create("Old Room")
        # backdate activity
        old = time.time() - 40 * 86400
        ctx.db.execute(
            "UPDATE rooms SET created_at=?, last_entered_at=? WHERE slug=?",
            (old, old, r.slug))
        stale = mgr.stale_rooms(days=30)
        self.assertTrue(any(x.slug == r.slug for x in stale))
        # still active — never auto-archived
        self.assertEqual(mgr.get(r.slug).status, "active")


class GoalRoomIntegrationTests(unittest.TestCase):
    def test_goal_step_runs_in_room(self):
        ctx, tmp = make_ctx(self)
        from nomorals.agents.goals import GoalSystem
        gs = GoalSystem(ctx)
        goal = gs.create("Room Goal", plan=["do the thing"])
        mgr = RoomManager(tmp, db=ctx.db)
        room = mgr.get_by_linked("goal", goal.id)
        self.assertIsNotNone(room)  # auto-created
        seen = []
        gs.advance(goal.id, executor=lambda d: seen.append(d) or "did it")
        self.assertEqual(seen, ["do the thing"])
        log = (Path(tmp) / "rooms" / room.slug / "logs" / "activity.log"
               ).read_text()
        self.assertIn("do the thing", log)

    def test_roomless_goal_byte_identical(self):
        ctx, tmp = make_ctx(self)
        ctx.settings.rooms_auto_create = False
        from nomorals.agents.goals import GoalSystem
        gs = GoalSystem(ctx)
        goal = gs.create("Plain Goal", plan=["plain step"])
        self.assertIsNone(
            RoomManager(tmp, db=ctx.db).get_by_linked("goal", goal.id))
        out = gs.advance(goal.id, executor=lambda d: "fine")
        step = [s for s in out.steps if s.description == "plain step"][0]
        self.assertEqual(step.status, "done")

    def test_auto_create_off(self):
        ctx, tmp = make_ctx(self)
        ctx.settings.rooms_auto_create = False
        from nomorals.agents.goals import GoalSystem
        gs = GoalSystem(ctx)
        goal = gs.create("No Room Goal", plan=["x"])
        self.assertIsNone(
            RoomManager(tmp, db=ctx.db).get_by_linked("goal", goal.id))


class ProjectRoomIntegrationTests(unittest.TestCase):
    def test_project_step_runs_in_room(self):
        ctx, tmp = make_ctx(self)
        from nomorals.agents.projects import ProjectManager
        pm = ProjectManager(ctx)
        p = pm.create("Room Project", steps=["build it"])
        mgr = RoomManager(tmp, db=ctx.db)
        room = mgr.get_by_linked("project", p.id)
        self.assertIsNotNone(room)  # auto-created
        out = pm.advance(p.id, executor=lambda d: "built")
        step = [s for s in out.steps if s.description == "build it"][0]
        self.assertEqual(step.status, "done")
        log = (Path(tmp) / "rooms" / room.slug / "logs" / "activity.log"
               ).read_text()
        self.assertIn("build it", log)

    def test_roomless_project_unchanged(self):
        ctx, tmp = make_ctx(self)
        ctx.settings.rooms_auto_create = False
        from nomorals.agents.projects import ProjectManager
        pm = ProjectManager(ctx)
        p = pm.create("Plain Project", steps=["plain"])
        out = pm.advance(p.id, executor=lambda d: "ok")
        self.assertEqual(out.steps[0].status, "done")


class SchedulerTests(unittest.TestCase):
    def test_ensure_rooms_tick_job_idempotent(self):
        ctx, tmp = make_ctx(self)
        from nomorals.workspace.rooms import ensure_rooms_tick_job
        first = ensure_rooms_tick_job(ctx)
        second = ensure_rooms_tick_job(ctx)
        self.assertTrue(first.get("scheduled") or
                        first.get("already_scheduled"))
        self.assertTrue(second.get("already_scheduled"))


if __name__ == "__main__":
    unittest.main()
