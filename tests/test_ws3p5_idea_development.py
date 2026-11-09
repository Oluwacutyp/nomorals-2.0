"""WS3 P5 — idea development: raw ideas grow, the raw text never moves.

Pins, per the pass brief:
(a) capture stores the raw text byte-identical and it survives a
    full development run unchanged;
(b) each of the four exits executes its real path (goal created in
    the named class / draft queued pending_review / mission persisted
    / parked record carries a trigger);
(c) research findings carry provenance or don't land;
(d) connections name real records (a seeded goal/idea is found; an
    absent one is not invented);
(e) the development record reads end-to-end and the verdict's
    reasoning is stored;
(f) ``nm idea`` CLI parity still works, plus the new develop action.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.idea_development import (
    DEFAULT_TRIGGER,
    Finding,
    IdeaDevelopment,
    render_record,
)
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.cmdline.commands.idea import _cmd_idea
from nomorals.core.config import load_settings
from nomorals.goals import GoalTracker, IdeaTracker
from nomorals.goals.surface import run_sync
from nomorals.missions.mission import MissionStore
from nomorals.social import drafts as drafts_module
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    help_text,
    parse_control,
)
from nomorals.social.drafts import DraftQueue
from nomorals.storage.db import Database

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

UID = "owner"


def _world() -> tuple[Database, IdeaDevelopment, IdeaTracker, GoalTracker]:
    db = Database(":memory:")
    db.migrate()  # the missions table rides migrations, as in production
    return db, IdeaDevelopment(db), IdeaTracker(db), GoalTracker(db)


def _expander(shape: str = "project", draft_text: str = ""):
    def _expand(idea: Any) -> dict[str, Any]:
        return {"what": "A shaped version of: " + idea.title,
                "why": "For the Ekiti builder community",
                "takes": "Two focused weeks",
                "unknowns": ["venue"], "shape": shape,
                "draft_text": draft_text, "suggested_class": ""}
    return _expand


def _research(findings: list[Finding]):
    def _run(idea: Any, expansion: Any, budget: int) -> list[Finding]:
        return list(findings)
    return _run


class _TmpQueue:
    """DraftQueue patched to a throwaway file for the duration."""

    def __init__(self, test: unittest.TestCase) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-ideas-p5-")
        self.queue = DraftQueue(db_path=str(Path(self._tmp.name) / "d.db"))
        self._real = drafts_module.DraftQueue
        drafts_module.DraftQueue = lambda *a, **k: self.queue  # type: ignore[misc]
        test.addCleanup(self._restore)

    def _restore(self) -> None:
        drafts_module.DraftQueue = self._real  # type: ignore[misc]
        self._tmp.cleanup()


class ImmutabilityTests(unittest.TestCase):
    def test_raw_text_survives_full_development(self) -> None:
        _db, cell, tracker, _goals = _world()
        queue = _TmpQueue(self).queue
        title = "Ekiti builders' radio — “quotes” & ünïcode"
        description = "Line one.\nLine two with trailing space \nEnd."
        idea = run_sync(tracker.create(UID, title,
                                       description=description))
        record = cell.develop(
            idea.idea_id, expand_fn=_expander(),
            research_fn=_research(
                [Finding("Study", "https://example.com/s")]),
            draft_queue=queue, mission_store=MissionStore(_db))
        self.assertEqual(record["status"], "completed")
        after = run_sync(tracker.get(idea.idea_id))
        assert after is not None
        self.assertEqual(after.title, title)
        self.assertEqual(after.description, description)

    def test_development_never_rewrites_the_idea_row(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "A quiet idea",
                                       description="stays as written"))

        def _forbidden(*a: Any, **k: Any) -> Any:
            raise AssertionError("development must not update the idea")

        cell.tracker.update = _forbidden  # type: ignore[method-assign]
        record = cell.develop(idea.idea_id, exit_override="park")
        self.assertEqual(record["status"], "completed")


class ExitTests(unittest.TestCase):
    def test_promote_in_named_class(self) -> None:
        _db, cell, tracker, goals = _world()
        idea = run_sync(tracker.create(UID, "Monthly portfolio review"))
        record = cell.develop(idea.idea_id, exit_override="promote",
                              class_arg="finance")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["verdict"]["class_id"], "finance")
        goal_id = record["outcome"]["goal_id"]
        goal = run_sync(goals.get(goal_id))
        assert goal is not None
        self.assertEqual(goal.goal_class, "finance")
        after = run_sync(tracker.get(idea.idea_id))
        assert after is not None
        self.assertEqual(after.status, "promoted_to_goal")
        self.assertEqual(after.metadata.get("promoted_goal_class"),
                         "finance")

    def test_promote_default_policy_files_general(self) -> None:
        _db, cell, tracker, goals = _world()
        idea = run_sync(tracker.create(UID, "Organise the spare room"))
        record = cell.develop(idea.idea_id)
        self.assertEqual(record["verdict"]["exit"], "promote")
        self.assertEqual(record["verdict"]["class_id"], "general")
        goal = run_sync(goals.get(record["outcome"]["goal_id"]))
        assert goal is not None
        self.assertEqual(goal.goal_class, "general")

    def test_draft_exit_lands_pending_review(self) -> None:
        _db, cell, tracker, _goals = _world()
        queue = _TmpQueue(self).queue
        idea = run_sync(tracker.create(UID, "Announce the clinic"))
        record = cell.develop(
            idea.idea_id,
            expand_fn=_expander(shape="content",
                                draft_text="The clinic opens Monday."),
            draft_queue=queue, mission_store=MissionStore(_db))
        self.assertEqual(record["verdict"]["exit"], "draft")
        self.assertEqual(record["status"], "completed")
        draft = queue.get(record["outcome"]["draft_id"])
        assert draft is not None
        self.assertEqual(draft.status, "pending_review")
        self.assertEqual(draft.content, "The clinic opens Monday.")
        self.assertEqual(draft.metadata.get("source"),
                         "idea-development")

    def test_mission_exit_persists_mission(self) -> None:
        db, cell, tracker, _goals = _world()
        missions = MissionStore(db)
        idea = run_sync(tracker.create(UID, "Survey the north farms"))
        record = cell.develop(idea.idea_id, exit_override="mission",
                              mission_store=missions)
        self.assertEqual(record["status"], "completed")
        mission = missions.get(record["outcome"]["mission_id"])
        self.assertIn("Survey the north farms", mission.goal)
        self.assertEqual(mission.state.get("idea_id"), idea.idea_id)
        after = run_sync(tracker.get(idea.idea_id))
        assert after is not None
        self.assertEqual(after.status, "active")

    def test_park_keeps_active_with_named_trigger(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "A studio someday"))
        record = cell.develop(idea.idea_id, exit_override="park",
                              trigger="when the studio lease renews")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["verdict"]["trigger"],
                         "when the studio lease renews")
        after = run_sync(tracker.get(idea.idea_id))
        assert after is not None
        self.assertEqual(after.status, "active")

    def test_park_without_trigger_gets_the_named_default(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Later, maybe"))
        record = cell.develop(idea.idea_id, exit_override="park")
        # a parked record with no trigger is a defect, never a state
        self.assertTrue(record["verdict"]["trigger"])
        self.assertEqual(record["verdict"]["trigger"], DEFAULT_TRIGGER)


class ResearchTests(unittest.TestCase):
    def test_unprovenanced_findings_never_land(self) -> None:
        db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Solar dryers for farms"))
        record = cell.develop(
            idea.idea_id, exit_override="park",
            research_fn=_research([
                Finding("Dryer study", "https://example.com/dryer"),
                Finding("No source at all", ""),
                Finding("Blank source", "   ")]))
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["research"]["refused"], 2)
        self.assertEqual(len(record["research"]["accepted"]), 1)
        claims = [c for c in cell.store.claims(record["contract_id"])
                  if c.kind == "evidence" and c.scope == "research"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].evidence_ref,
                         "https://example.com/dryer")
        self.assertEqual(claims[0].verdict, "supported")

    def test_no_seam_records_the_statement_not_a_fake(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Offline thought"))
        record = cell.develop(idea.idea_id, exit_override="park",
                              research_fn=None)
        self.assertEqual(record["status"], "completed")
        self.assertIn("no research seam",
                      record["research"]["statement"])

    def test_research_failure_is_named_in_the_record(self) -> None:
        _db, cell, tracker, _goals = _world()

        def _boom(idea: Any, expansion: Any, budget: int) -> list[Finding]:
            raise RuntimeError("search backend down")

        idea = run_sync(tracker.create(UID, "Fragile lookup"))
        record = cell.develop(idea.idea_id, exit_override="park",
                              research_fn=_boom)
        self.assertEqual(record["status"], "completed")
        self.assertIn("research step failed",
                      record["research"]["statement"])
        self.assertIn("search backend down",
                      record["research"]["statement"])


class ConnectionTests(unittest.TestCase):
    def test_seeded_connections_are_found_and_named(self) -> None:
        _db, cell, tracker, goals = _world()
        seeded = run_sync(goals.create(
            UID, "Grow the careers practice", goal_class="careers"))
        idea = run_sync(tracker.create(
            UID, "Weekly careers clinic for Ekiti builders",
            description="Helping builders prep CVs and interviews."))
        kin = run_sync(tracker.create(
            UID, "Careers clinic office hours",
            description="Weekly careers help sessions for builders."))
        record = cell.develop(idea.idea_id, exit_override="park")
        connections = record["connections"]
        self.assertEqual(connections["goal_class"]["id"], "careers")
        named_goals = {g["goal_id"] for g in connections["class_goals"]}
        self.assertIn(seeded.goal_id, named_goals)
        named_kin = {k["idea_id"] for k in connections["kin_ideas"]}
        self.assertIn(kin.idea_id, named_kin)

    def test_absent_connections_are_not_invented(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(
            UID, "Zqxwv unique widget calibration",
            description="Nothing else in the pile resembles this."))
        record = cell.develop(idea.idea_id, exit_override="park")
        connections = record["connections"]
        self.assertEqual(connections["class_goals"], [])
        self.assertEqual(connections["kin_ideas"], [])
        self.assertEqual(connections["projects"], [])

    def test_project_named_only_on_mention(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(
            UID, "GoldenSwing journal export",
            description="Pull the journal into a weekly digest."))
        record = cell.develop(idea.idea_id, exit_override="park")
        self.assertEqual(record["connections"]["projects"],
                         ["GoldenSwing"])

    def test_unknown_class_refused_before_any_run(self) -> None:
        db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Something"))
        with self.assertRaises(ValueError):
            cell.develop(idea.idea_id, exit_override="promote",
                         class_arg="no-such-class")
        rows = db.query("SELECT * FROM idea_development WHERE idea_id = ?",
                        (idea.idea_id,))
        self.assertEqual(rows, [])


class RecordTests(unittest.TestCase):
    def test_record_reads_end_to_end_and_reasoning_is_stored(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "A reading circle"))
        record = cell.develop(idea.idea_id, exit_override="park",
                              trigger="when three members commit")
        self.assertTrue(record["verdict"]["reasoning"])
        stored = cell.record_for(record["dev_id"])
        self.assertEqual(stored, record)
        latest = cell.latest_for(idea.idea_id)
        assert latest is not None
        self.assertEqual(latest["dev_id"], record["dev_id"])
        text = render_record(record)
        for needle in ("A reading circle", "expansion", "connections",
                       "research", "verdict: park",
                       "the owner named the exit: park",
                       "revival trigger: when three members commit",
                       "verifier", "outcome: completed"):
            self.assertIn(needle, text)
        # internal ids never print in the rendered record
        self.assertNotIn(record["dev_id"], text)
        self.assertNotIn(record["contract_id"], text)
        self.assertNotIn(idea.idea_id, text)

    def test_second_development_fences_forward(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Develop me twice"))
        first = cell.develop(idea.idea_id, exit_override="park")
        second = cell.develop(idea.idea_id, exit_override="park")
        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)
        self.assertEqual(len(cell.developments_for(idea.idea_id)), 2)
        artifact = cell.store.artifact("idea-dev:" + idea.idea_id)
        assert artifact is not None
        self.assertEqual(artifact["revision"], 2)


class GateTests(unittest.TestCase):
    def test_tampered_raw_text_blocks_the_exit(self) -> None:
        _db, cell, tracker, goals = _world()
        idea = run_sync(tracker.create(UID, "Honest capture"))

        def _tamper(target: Any) -> dict[str, Any]:
            run_sync(cell.tracker.update(
                target.idea_id, title="rewritten behind the run"))
            return _expander()(target)

        record = cell.develop(idea.idea_id, expand_fn=_tamper)
        self.assertEqual(record["status"], "failed")
        self.assertIn("verification gate refused",
                      record["outcome"]["detail"])
        checks = {c["name"]: c["ok"]
                  for c in record["verifier"]["checks"]}
        self.assertFalse(checks["raw_unchanged"])
        self.assertEqual(run_sync(goals.list_goals(UID)), [])
        contract = cell.store.get_contract(record["contract_id"])
        assert contract is not None
        self.assertNotEqual(contract.status, "completed")

    def test_verifier_is_never_the_worker(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Separation of duties"))
        record = cell.develop(idea.idea_id, exit_override="park")
        contract = cell.store.get_contract(record["contract_id"])
        assert contract is not None
        self.assertNotEqual(contract.worker, contract.verifier)
        for claim in cell.store.claims(record["contract_id"]):
            if claim.kind == "evidence":
                self.assertNotEqual(claim.validator, claim.producer)

    def test_dismissed_and_promoted_ideas_are_refused_in_words(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Dismiss me"))
        run_sync(tracker.dismiss(idea.idea_id, reason="no"))
        with self.assertRaises(ValueError):
            cell.develop(idea.idea_id)
        other = run_sync(tracker.create(UID, "Promote me once"))
        cell.develop(other.idea_id)
        with self.assertRaises(ValueError):
            cell.develop(other.idea_id)


class RescoreTests(unittest.TestCase):
    def test_unscored_idea_gets_no_invented_score(self) -> None:
        _db, cell, tracker, _goals = _world()
        idea = run_sync(tracker.create(UID, "Unscored thought"))
        record = cell.develop(idea.idea_id, exit_override="park")
        self.assertIn("does not invent one", record["ice"]["note"])
        after = run_sync(tracker.get(idea.idea_id))
        assert after is not None
        self.assertIsNone(after.ice_score)

    def test_scored_idea_adjustments_are_named(self) -> None:
        _db, cell, tracker, goals = _world()
        run_sync(goals.create(UID, "Grow the careers practice",
                              goal_class="careers"))
        idea = run_sync(tracker.create(
            UID, "Weekly careers clinic for Ekiti builders",
            description="Helping builders prep CVs and interviews."))
        run_sync(tracker.create(
            UID, "Careers clinic office hours",
            description="Weekly careers help sessions for builders."))
        run_sync(tracker.score_idea(idea.idea_id, impact=5.0,
                                    confidence=5.0, ease=5.0))
        record = cell.develop(
            idea.idea_id,
            research_fn=_research([
                Finding("One", "https://example.com/1"),
                Finding("Two", "https://example.com/2")]))
        after = run_sync(tracker.get(idea.idea_id))
        assert after is not None
        self.assertEqual(after.impact, 6.0)
        self.assertEqual(after.confidence, 7.0)
        self.assertEqual(after.ease, 5.0)
        named = " ".join(record["ice"]["adjustments"])
        self.assertIn("kin support", named)
        self.assertIn("provenanced research", named)
        self.assertIn("class traction", named)


class BrainToolTests(unittest.TestCase):
    def test_capture_and_develop_through_the_brain_tool(self) -> None:
        _TmpQueue(self)
        db = Database(":memory:")
        db.migrate()
        captured: dict[str, Any] = {}

        class _Registry:
            context = SimpleNamespace(db=db, tools=None)

            def register(self, name: str, **kwargs: Any):
                def _wrap(fn: Any) -> Any:
                    captured[name] = fn
                    return fn
                return _wrap

        from nomorals.agents.idea_development import register

        register(_Registry())
        idea_tool = captured["idea"]
        out = idea_tool("capture",
                        text="A weekend market for Ekiti makers")
        self.assertTrue(out["ok"])
        listed = idea_tool("list")
        self.assertEqual([i["title"] for i in listed["ideas"]],
                         ["A weekend market for Ekiti makers"])
        developed = idea_tool("develop", ref="#1", exit="park",
                              trigger="when the hall is booked")
        self.assertTrue(developed["ok"])
        self.assertEqual(developed["exit"], "park")
        self.assertIn("revival trigger: when the hall is booked",
                      developed["record"])


class CatalogTests(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("ideas", CONTROL_COMMANDS)
        self.assertIn("ideas", COMMAND_DETAILS)
        self.assertIn("ideas", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("ideas"), 1)
        self.assertIn("/ideas", COMMAND_DETAILS["ideas"]["usage"])
        self.assertTrue(COMMAND_DETAILS["ideas"]["what"])
        self.assertTrue(COMMAND_DETAILS["ideas"]["example"])
        self.assertIn("/ideas", help_text())
        self.assertIn("/ideas", detailed_help("ideas"))

    def test_owner_only(self) -> None:
        self.assertNotIn("ideas", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("ideas", False))
        self.assertTrue(caller_may_run("ideas", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/ideas develop 1")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "ideas")
        self.assertIsNotNone(parse_control("/ideas"))


class ChatSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ideas-p5-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        reply = self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999"))
        return reply.text if hasattr(reply, "text") else reply

    def test_capture_list_and_park_via_chat(self) -> None:
        out = self._run("/ideas add A radio show for Ekiti builders"
                        " | Weekly interviews with makers")
        self.assertIn("captured", out)
        board = self._run("/ideas")
        self.assertIn("A radio show for Ekiti builders", board)
        out = self._run("/ideas develop 1 as park"
                        " when the studio is free")
        self.assertIn("parked", out)
        # the chat grammar is [when <trigger>] — the trigger itself
        # is the words after "when"
        self.assertIn("revival trigger: the studio is free", out)
        tracker = IdeaTracker(self.context.db)
        ideas = run_sync(tracker.list_ideas(UID))
        self.assertEqual(ideas[0].title,
                         "A radio show for Ekiti builders")
        self.assertEqual(ideas[0].description,
                         "Weekly interviews with makers")
        self.assertEqual(ideas[0].status, "active")

    def test_develop_promotes_in_class_via_chat(self) -> None:
        self._run("/ideas add Fund the tool library")
        out = self._run("/ideas develop 1 as goal in finance")
        self.assertIn("promoted", out)
        self.assertIn("Finance", out)
        goals = run_sync(GoalTracker(self.context.db).list_goals(
            UID, goal_class="finance"))
        self.assertEqual([g.title for g in goals],
                         ["Fund the tool library"])
        shown = self._run("/ideas show 1")
        self.assertIn("verdict: promote", shown)


def _cli_ns(**overrides: object) -> argparse.Namespace:
    base = dict(action="list", arg="", arg2="", id="", title="",
                description="", priority="", project=False, remove=False,
                reason="", filter_status="", limit="20", tags="",
                source="", user="owner", json=False,
                exit_override="", goal_class="", trigger="")
    base.update(overrides)
    return argparse.Namespace(**base)


def _cli(args: argparse.Namespace, ctx: Any) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _cmd_idea(args, ctx)
    return code, buf.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        _TmpQueue(self)
        db = Database(":memory:")
        db.migrate()
        self.ctx = SimpleNamespace(
            db=db,
            settings=SimpleNamespace(rooms_auto_create=False,
                                     autonomy=None),
            router=None)

    def test_create_list_promote_parity(self) -> None:
        code, out = _cli(_cli_ns(action="create", arg="Parity idea",
                                  description="d1"), self.ctx)
        self.assertEqual(code, 0)
        self.assertIn("created", out)
        idea_id = out.split()[1]
        code, out = _cli(_cli_ns(action="list"), self.ctx)
        self.assertEqual(code, 0)
        self.assertIn("Parity idea", out)
        code, out = _cli(_cli_ns(action="promote", id=idea_id), self.ctx)
        self.assertEqual(code, 0)
        self.assertIn("promoted", out)

    def test_develop_action_parks_with_trigger(self) -> None:
        code, out = _cli(_cli_ns(action="create", arg="CLI grown idea"),
                         self.ctx)
        idea_id = out.split()[1]
        code, out = _cli(_cli_ns(action="develop", id=idea_id,
                                  exit_override="park",
                                  trigger="when the CLI says so"),
                         self.ctx)
        self.assertEqual(code, 0)
        self.assertIn("parked", out)
        self.assertIn("when the CLI says so", out)
        code, out = _cli(_cli_ns(action="develop", id="idea-missing"),
                         self.ctx)
        self.assertEqual(code, 1)
        self.assertIn("not found", out)


if __name__ == "__main__":
    unittest.main()
