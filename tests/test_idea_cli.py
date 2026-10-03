"""CLI coverage: ``nm idea`` plus the new ``nm goal`` update/delete/replan.

_nm idea_ is the command surface for nomorals/goals.IdeaTracker; the goal
actions exercise GoalSystem.update / delete / replan, which the parser
advertised but the dispatcher never implemented.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace

from nomorals.cmdline.commands.idea import _cmd_idea
from nomorals.cmdline.commands.goal import _cmd_goal
from nomorals.storage.db import Database


def _ctx() -> SimpleNamespace:
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(rooms_auto_create=False, autonomy=None),
        router=None,
    )


def _ns(**overrides: object) -> argparse.Namespace:
    base = dict(action="list", arg="", arg2="", id="", title="",
                description="", priority="", project=False, remove=False,
                reason="", filter_status="", limit="20", tags="",
                source="", user="owner", json=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _run(fn, args, ctx) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(args, ctx)
    return code, buf.getvalue()


class IdeaCliTests(unittest.TestCase):
    def test_create_list_get(self):
        ctx = _ctx()
        code, out = _run(_cmd_idea, _ns(action="create", arg="My idea",
                                       description="d1", tags="a,b"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("created", out)
        idea_id = out.split()[1]

        code, out = _run(_cmd_idea, _ns(action="list"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("My idea", out)

        code, out = _run(_cmd_idea, _ns(action="get", id=idea_id), ctx)
        self.assertEqual(code, 0)
        self.assertIn("My idea", out)

    def test_get_missing(self):
        code, out = _run(_cmd_idea, _ns(action="get", id="idea-nope"), _ctx())
        self.assertEqual(code, 1)
        self.assertIn("not found", out)

    def test_update_dismiss(self):
        ctx = _ctx()
        _, out = _run(_cmd_idea, _ns(action="create", arg="T"), ctx)
        idea_id = out.split()[1]
        code, out = _run(_cmd_idea, _ns(action="update", id=idea_id,
                                        title="T2"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("updated", out)
        code, out = _run(_cmd_idea, _ns(action="dismiss", id=idea_id,
                                        reason="later"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("dismissed", out)
        code, _ = _run(_cmd_idea, _ns(action="dismiss", id="idea-nope"), ctx)
        self.assertEqual(code, 1)

    def test_promote_and_delete(self):
        ctx = _ctx()
        _, out = _run(_cmd_idea, _ns(action="create", arg="Promote me",
                                     tags="x"), ctx)
        idea_id = out.split()[1]
        code, out = _run(_cmd_idea, _ns(action="promote", id=idea_id), ctx)
        self.assertEqual(code, 0)
        self.assertIn("promoted", out)
        self.assertIn("goal", out)
        code, _ = _run(_cmd_idea, _ns(action="promote", id="idea-nope"), ctx)
        self.assertEqual(code, 1)

        _, out = _run(_cmd_idea, _ns(action="create", arg="Gone"), ctx)
        gone_id = out.split()[1]
        code, out = _run(_cmd_idea, _ns(action="delete", id=gone_id), ctx)
        self.assertEqual(code, 0)
        self.assertIn("deleted", out)

    def test_search_and_stats(self):
        ctx = _ctx()
        _run(_cmd_idea, _ns(action="create", arg="Space game"), ctx)
        code, out = _run(_cmd_idea, _ns(action="search", arg="space"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("Space game", out)
        code, out = _run(_cmd_idea, _ns(action="stats"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("ideas: 1", out)

    def test_unknown_action(self):
        code, _ = _run(_cmd_idea, _ns(action="frobnicate"), _ctx())
        self.assertEqual(code, 2)

    def test_json_output(self):
        ctx = _ctx()
        _run(_cmd_idea, _ns(action="create", arg="J"), ctx)
        code, out = _run(_cmd_idea, _ns(action="list", json=True), ctx)
        self.assertEqual(code, 0)
        import json as _json
        payload = _json.loads(out)
        self.assertEqual(len(payload["ideas"]), 1)


class GoalNewActionsTests(unittest.TestCase):
    def test_goal_update_action(self):
        ctx = _ctx()
        code, out = _run(_cmd_goal, _ns(action="create", arg="Old title"), ctx)
        self.assertEqual(code, 0)
        goal_id = out.split()[1]
        code, out = _run(_cmd_goal, _ns(action="update", id=goal_id,
                                        title="New title", priority="3"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("updated", out)
        code, out = _run(_cmd_goal, _ns(action="get", id=goal_id), ctx)
        self.assertIn("New title", out)
        code, _ = _run(_cmd_goal, _ns(action="update", id="goal-nope",
                                       title="x"), ctx)
        self.assertEqual(code, 1)

    def test_goal_delete_action(self):
        ctx = _ctx()
        _, out = _run(_cmd_goal, _ns(action="create", arg="Doomed"), ctx)
        goal_id = out.split()[1]
        code, out = _run(_cmd_goal, _ns(action="delete", id=goal_id), ctx)
        self.assertEqual(code, 0)
        self.assertIn("deleted", out)
        code, out = _run(_cmd_goal, _ns(action="get", id=goal_id), ctx)
        self.assertEqual(code, 1)
        code, _ = _run(_cmd_goal, _ns(action="delete", id="goal-nope"), ctx)
        self.assertEqual(code, 1)

    def test_goal_replan_action(self):
        ctx = _ctx()
        _, out = _run(_cmd_goal, _ns(action="create", arg="Replan me"), ctx)
        goal_id = out.split()[1]
        code, out = _run(_cmd_goal, _ns(action="replan", id=goal_id,
                                        reason="changed mind"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("replanned", out)
        code, out = _run(_cmd_goal, _ns(action="replan", id="goal-nope"), ctx)
        self.assertEqual(code, 1)


class GoalSystemMethodTests(unittest.TestCase):
    def _gs(self):
        from nomorals.agents.goals import GoalSystem
        ctx = _ctx()
        return GoalSystem(ctx)

    def test_update_method(self):
        gs = self._gs()
        g = gs.create("T", plan=["a", "b"])
        updated = gs.update(g.id, title="T2", description="d2", priority=7)
        assert updated is not None
        self.assertEqual(updated.title, "T2")
        self.assertEqual(updated.description, "d2")
        self.assertEqual(updated.priority, 7)
        self.assertIsNone(gs.update("goal-nope", title="x"))
        with self.assertRaises(ValueError):
            gs.update(g.id, title="  ")

    def test_delete_method_scrubs_dependencies(self):
        gs = self._gs()
        a = gs.create("A", plan=["a1"])
        b = gs.create("B", plan=["b1"])
        gs.add_dependency(b.id, a.id)
        self.assertTrue(gs.delete(a.id))
        self.assertIsNone(gs.get(a.id))
        self.assertEqual(gs.get(b.id).depends_on, [])
        self.assertFalse(gs.delete(a.id))

    def test_replan_method(self):
        gs = self._gs()
        g = gs.create("T", plan=["one", "two", "three"])
        replanned = gs.replan(g.id, reason="test")
        assert replanned is not None
        self.assertTrue(len(replanned.steps) >= 3)
        self.assertIsNone(gs.replan("goal-nope"))


class IdeaParserTests(unittest.TestCase):
    def test_idea_subcommand_parses(self):
        from nomorals.cmdline.parser import _parser
        p = _parser()
        args = p.parse_args(["idea", "create", "hello"])
        self.assertEqual(args.command, "idea")
        self.assertEqual(args.action, "create")
        self.assertEqual(args.arg, "hello")
        args = p.parse_args(["ideas", "list", "--status", "active"])
        self.assertEqual(args.command, "ideas")
        self.assertEqual(args.action, "list")

    def test_goal_actions_include_new_verbs(self):
        from nomorals.cmdline.parser import _parser
        p = _parser()
        args = p.parse_args(["goal", "replan", "goal-1"])
        self.assertEqual(args.action, "replan")


if __name__ == "__main__":
    unittest.main()
