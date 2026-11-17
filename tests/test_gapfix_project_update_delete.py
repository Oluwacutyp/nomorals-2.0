"""Failing-first: `nm project update` / `nm project delete` (audit R2-41).

The parser advertises both verbs; the handler refuses them because
ProjectManager has no update/delete operation. These tests pin the
real behaviour: update changes exactly the named field (status,
history and journal entries survive); delete removes the project
row + steps KV + integrity-mode KV, keeps the frozen history, and
never orphans a dependent silently.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.projects import ProjectManager
from nomorals.storage.db import Database
from nomorals.storage.kv import KVStore


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _ProjectBase(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(
            dir=os.environ.get("NM_TEST_TMPDIR") or None))
        self.addCleanup(lambda: shutil.rmtree(self.workdir, True))
        self.db = _db()
        ctx = SimpleNamespace(
            db=self.db, home=self.workdir,
            settings=SimpleNamespace(workspace_dir=str(self.workdir)))
        self.pm = ProjectManager(ctx)
        self.ctx = ctx

    def _make(self, title="plain", objective="do a thing", steps=None):
        return self.pm.create(title, objective=objective,
                              steps=steps or ["step one", "step two"])

    def _args(self, action, title="", description=""):
        return SimpleNamespace(action=action, title=title,
                               description=description, id="", json=False)


class UpdateTests(_ProjectBase):
    def test_update_title_changes_only_title(self):
        p = self._make()
        self.pm._record_acceptance_run(p, "s1", "true", "ok")
        before = self.pm._load(p.id)
        assert before is not None
        res = self.pm.update(p.id, title="Renamed")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res.get("updated", {}).get("title"), "Renamed")
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.title, "Renamed")
        self.assertEqual(after.objective, before.objective)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.progress, before.progress)
        self.assertEqual([s.description for s in after.steps],
                         [s.description for s in before.steps])
        # frozen history survives the update
        self.assertEqual(len(self.pm.acceptance_history(p.id)), 1)

    def test_update_objective_does_not_reclassify(self):
        p = self._make(title="widget", objective="build a widget")
        before = self.pm._load(p.id)
        assert before is not None
        res = self.pm.update(p.id, objective="research the market")
        self.assertTrue(res.get("ok"), res)
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.objective, "research the market")
        # only the named field changes — no silent re-derivation
        self.assertEqual(after.task_kind, before.task_kind)
        self.assertEqual(after.artifact, before.artifact)
        self.assertEqual(after.verify_cmd, before.verify_cmd)

    def test_update_budget_coerces(self):
        p = self._make()
        res = self.pm.update(p.id, budget_wall="12.5", budget_tokens="100")
        self.assertTrue(res.get("ok"), res)
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.budget_wall, 12.5)
        self.assertEqual(after.budget_tokens, 100)

    def test_update_unknown_field_refused(self):
        p = self._make()
        res = self.pm.update(p.id, frobnicate="x")
        self.assertFalse(res.get("ok"))
        self.assertIn("frobnicate", str(res.get("error", "")))
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.title, "plain")

    def test_update_status_refused(self):
        p = self._make()
        res = self.pm.update(p.id, status="done")
        self.assertFalse(res.get("ok"))
        self.assertIn("status", str(res.get("error", "")))
        after = self.pm._load(p.id)
        assert after is not None
        self.assertNotEqual(after.status, "done")

    def test_update_unknown_project_refused(self):
        res = self.pm.update("no-such-project", title="x")
        self.assertFalse(res.get("ok"))
        self.assertIn("no-such-project", str(res.get("error", "")))

    def test_update_bad_budget_value_refused(self):
        p = self._make()
        res = self.pm.update(p.id, budget_wall="not-a-number")
        self.assertFalse(res.get("ok"))
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.budget_wall, 0.0)

    def test_update_bad_task_kind_refused(self):
        p = self._make()
        res = self.pm.update(p.id, task_kind="teleport")
        self.assertFalse(res.get("ok"))
        self.assertIn("teleport", str(res.get("error", "")))


class DeleteTests(_ProjectBase):
    def test_delete_removes_row_steps_and_integrity_mode(self):
        from nomorals.agents.auditor import get_integrity_mode, set_integrity_mode
        p = self._make()
        self.pm._record_acceptance_run(p, "s1", "true", "ok")
        set_integrity_mode(self.db, p.id, "demo")
        self.assertEqual(get_integrity_mode(self.db, p.id), "demo")
        res = self.pm.delete(p.id)
        self.assertTrue(res.get("ok"), res)
        self.assertIsNone(self.pm._load(p.id))
        self.assertEqual(self.pm.list_projects(), [])
        self.assertIsNone(KVStore(self.db).get("project.steps." + p.id))
        # integrity mode is config, not journal — it goes with the project
        self.assertEqual(get_integrity_mode(self.db, p.id), "development")
        # frozen history survives the delete
        self.assertEqual(len(self.pm.acceptance_history(p.id)), 1)
        self.assertIn("project row", " ".join(res.get("removed", [])))
        retained = res.get("retained", {})
        self.assertEqual(retained.get("acceptance_runs"), 1)

    def test_delete_second_time_refuses(self):
        p = self._make()
        self.assertTrue(self.pm.delete(p.id).get("ok"))
        res = self.pm.delete(p.id)
        self.assertFalse(res.get("ok"))
        self.assertIn(p.id, str(res.get("error", "")))

    def test_delete_unknown_project_refuses(self):
        res = self.pm.delete("no-such-project")
        self.assertFalse(res.get("ok"))
        self.assertIn("no-such-project", str(res.get("error", "")))

    def test_delete_running_project_completes_and_names_dependents(self):
        # No durable workflow/mission run is ever linked to a project
        # (nothing in the repo references a project id from a run), so
        # delete completes cleanly and names what it leaves in place.
        from nomorals.workspace.rooms import RoomManager
        p = self._make(steps=["build it"])
        self.assertEqual(p.status, "running")
        # a linked goal (mirrored, never owned) and a linked room
        self.db.execute(
            "INSERT INTO agent_goals (id, title, status, project_id) "
            "VALUES (?,?,?,?)", ("goal-1", "g", "active", p.id))
        p.goal_id = "goal-1"
        self.pm._upsert_row(p)
        room = RoomManager(self.workdir, db=self.db).get_by_linked(
            "project", p.id)
        self.assertIsNotNone(room)
        res = self.pm.delete(p.id)
        self.assertTrue(res.get("ok"), res)
        retained = res.get("retained", {})
        self.assertEqual(retained.get("linked_goal"), "goal-1")
        self.assertTrue(retained.get("linked_room"))
        # the goal row and the room both survive — they belong to
        # other organs, not to the project
        row = self.db.query_one(
            "SELECT id FROM agent_goals WHERE id=?", ("goal-1",))
        self.assertIsNotNone(row)
        self.assertIsNotNone(
            RoomManager(self.workdir, db=self.db).get_by_linked(
                "project", p.id))


class CliTests(_ProjectBase):
    def test_cli_update_field_equals(self):
        from nomorals.cmdline.commands.project import _cmd_project
        p = self._make()
        rc = _cmd_project(self._args("update", p.id, "title=Renamed"), self.ctx)
        self.assertEqual(rc, 0)
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.title, "Renamed")

    def test_cli_update_field_space(self):
        from nomorals.cmdline.commands.project import _cmd_project
        p = self._make()
        rc = _cmd_project(
            self._args("update", p.id, "title Renamed Title"), self.ctx)
        self.assertEqual(rc, 0)
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.title, "Renamed Title")

    def test_cli_update_unknown_project_rc1(self):
        from nomorals.cmdline.commands.project import _cmd_project
        rc = _cmd_project(
            self._args("update", "no-such", "title=x"), self.ctx)
        self.assertEqual(rc, 1)

    def test_cli_update_missing_field_value_rc1(self):
        from nomorals.cmdline.commands.project import _cmd_project
        p = self._make()
        rc = _cmd_project(self._args("update", p.id, ""), self.ctx)
        self.assertEqual(rc, 1)
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.title, "plain")

    def test_cli_delete_rc0_and_second_rc1(self):
        from nomorals.cmdline.commands.project import _cmd_project
        p = self._make()
        self.assertEqual(_cmd_project(self._args("delete", p.id), self.ctx), 0)
        self.assertIsNone(self.pm._load(p.id))
        self.assertEqual(_cmd_project(self._args("delete", p.id), self.ctx), 1)

    def test_cli_delete_unknown_rc1(self):
        from nomorals.cmdline.commands.project import _cmd_project
        rc = _cmd_project(self._args("delete", "no-such"), self.ctx)
        self.assertEqual(rc, 1)


class ToolTests(_ProjectBase):
    def _tool(self):
        from nomorals.agents.projects import register

        class _Reg:
            def __init__(self, ctx):
                self.context = ctx
                self.tools = {}

            def register(self, name, **kwargs):
                def deco(fn):
                    self.tools[name] = fn
                    return fn
                return deco

        reg = _Reg(self.ctx)
        register(reg)
        return reg.tools["project"]

    def test_tool_update_and_delete(self):
        tool = self._tool()
        p = self._make()
        res = tool("update", project_id=p.id, field="title", value="Via tool")
        self.assertTrue(res.get("ok"), res)
        after = self.pm._load(p.id)
        assert after is not None
        self.assertEqual(after.title, "Via tool")
        res = tool("delete", project_id=p.id)
        self.assertTrue(res.get("ok"), res)
        self.assertIsNone(self.pm._load(p.id))


if __name__ == "__main__":
    unittest.main()
