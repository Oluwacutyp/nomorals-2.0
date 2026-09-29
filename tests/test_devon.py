"""Devon: the autonomous dev agent (/devon) and the games-in-every-chat rule.

Offline by design: a scripted router stands in for the planner and digest
LLM, and the keyword heuristic covers the no-model path. Coverage:

* parsing — ``/devon [free text]`` is a control command; help mentions it
* the planner — LLM-JSON plan (unknown tools dropped) and the keyword
  heuristic (message flow / tests / research / mood / default)
* every tool — including the refusals (path traversal, non-SELECT SQL)
* the memory box — steps + digest persisted per run, digests carried
  forward into the next planner prompt
* runtime wiring — /devon is owner-only; a member's slash is just text
* games anywhere — a member of ANY chat can start a game and play moves;
  every other slash command stays owner-only
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path

from tests.test_partner_runtime import FakeAdapter, FakeRouter, _make_context, _wait

from nomorals.agents.devon import TOOL_CATALOG, DevonAgent
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import help_text, parse_control
from nomorals.social.chat.gateway import ChatGateway


# ── helpers ──────────────────────────────────────────────────────────────────


class DevonRouter(FakeRouter):
    """FakeRouter that can also script a valid planner plan + digest."""

    def __init__(self, plan: list[dict] | None = None, digest: str = "",
                 replies: list[str] | None = None) -> None:
        super().__init__(replies=replies)
        self.plan = plan
        self.digest = digest
        self.planner_prompts: list[str] = []

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw) -> LLMResponse:
        blob = " ".join(m.content for m in messages)
        if "autonomous engineering & debug agent" in blob:
            self.planner_prompts.append(blob)
            if self.plan is not None:
                return LLMResponse(text=json.dumps({"steps": self.plan}), model="fake-planner")
            return LLMResponse(text="(no plan)", model="fake-planner")
        if "Summarize the tool output" in blob and self.digest:
            return LLMResponse(text=self.digest, model="fake-digest")
        return super().chat(messages, params, **kw)


# ── parsing ──────────────────────────────────────────────────────────────────


class DevonParseTest(unittest.TestCase):
    def test_devon_parses_with_task(self) -> None:
        cmd = parse_control("/devon check if the brain replied to the last messages")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "devon")
        self.assertIn("brain replied", cmd.tail)

    def test_devon_bare(self) -> None:
        cmd = parse_control("/devon")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "devon")
        self.assertEqual(cmd.tail, "")

    def test_devon_in_help(self) -> None:
        self.assertIn("/devon", help_text())

    def test_help_covers_devon(self) -> None:
        from nomorals.social.chat.control import CONTROL_COMMANDS

        self.assertIn("devon", CONTROL_COMMANDS)
        self.assertIn("/devon", help_text())


# ── heuristic planner ────────────────────────────────────────────────────────


class DevonHeuristicPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.agent = DevonAgent(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _tools(self, task: str) -> list[str]:
        return [s["tool"] for s in self.agent._heuristic_plan(task)]

    def test_user_example_task_hits_message_flow(self) -> None:
        tools = self._tools(
            "check the current chatbot workflow, and see if the messages "
            "sent to brain got replied directly by the brain and drops into chat"
        )
        self.assertIn("message_flow", tools)
        self.assertIn("read_code", tools)

    def test_tests_task(self) -> None:
        self.assertIn("run_tests", self._tools("are the tests passing? anything broken?"))

    def test_research_task(self) -> None:
        self.assertIn("research", self._tools("look up what is llama.cpp"))

    def test_mood_task(self) -> None:
        self.assertIn("mood", self._tools("how is your mood right now"))

    def test_logs_task(self) -> None:
        tools = self._tools("what does the log say about the last crash")
        self.assertIn("logs_tail", tools)

    def test_default_task_orients(self) -> None:
        tools = self._tools("what are you up to")
        self.assertIn("git_status", tools)

    def test_plan_never_uses_unknown_tools(self) -> None:
        valid = {name for name, _ in TOOL_CATALOG}
        for task in ("check the message flow", "run the tests", "research x",
                     "what's going on", "tail the logs", "start a mission to watch the queue"):
            for step in self.agent._heuristic_plan(task):
                self.assertIn(step["tool"], valid, f"{task!r} planned unknown tool")

    def test_mission_task(self) -> None:
        tools = self._tools("start a background job to build the exporter")
        self.assertIn("mission", tools)

    def test_monitoring_task_is_a_watch(self) -> None:
        # Monitoring = a scheduled re-check (watch), not a one-shot mission.
        tools = self._tools("keep monitoring the queue every hour")
        self.assertIn("watch", tools)
        self.assertNotIn("mission", tools)

    def test_uncensored_question_does_not_run_tests(self) -> None:
        # The phone caught this: "uncensored" contains the substring "red",
        # so the old matcher launched the full test suite for an identity
        # question. Word-boundary matching must fix it.
        tools = self._tools("are you truly uncensored?")
        self.assertNotIn("run_tests", tools)
        self.assertIn("read_code", tools)  # answer from the persona source

    def test_word_boundary_matching(self) -> None:
        # Substring traps: none of these words may fire their keyword.
        self.assertNotIn("run_tests", self._tools("please censored content check"))   # "red"
        self.assertNotIn("find_code", self._tools("what were you already doing"))     # "read"
        self.assertNotIn("chat_stats", self._tools("how to migrate the database"))    # "rate"
        # …and the real keywords still work.
        self.assertIn("run_tests", self._tools("are the tests failing?"))
        self.assertIn("research", self._tools("what is llama.cpp"))
        self.assertIn("run_tests", self._tools("the suite is red again"))

    def test_new_tool_routing(self) -> None:
        self.assertIn("models", self._tools("which models are configured, is hf working?"))
        self.assertIn("env", self._tools("what is my config, is power mode on?"))
        self.assertIn("history", self._tools("what did you find last time about the arena?"))
        self.assertIn("trace", self._tools("trace the last conversation"))
        self.assertIn("git_diff", self._tools("diff between the last two commits"))
        self.assertIn("chat_stats", self._tools("any live games or platform stats?"))

    def test_watch_interval_parsed_from_task(self) -> None:
        steps = self.agent._heuristic_plan("watch the queue every 30 minutes")
        watch = [s for s in steps if s["tool"] == "watch"]
        self.assertTrue(watch, f"no watch step in {steps}")
        self.assertEqual(watch[0]["args"]["interval_minutes"], 30)


# ── tools ────────────────────────────────────────────────────────────────────


class DevonToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.agent = DevonAgent(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_read_code_with_grep(self) -> None:
        out = self.agent._tool_read_code(
            {"path": "nomorals/agents/devon.py", "grep": "class DevonAgent"}
        )
        self.assertIn("class DevonAgent", out)

    def test_read_code_refuses_traversal(self) -> None:
        out = self.agent._tool_read_code({"path": "../../etc/passwd"})
        self.assertIn("no such file", out)

    def test_read_code_missing_path(self) -> None:
        self.assertIn("no such file", self.agent._tool_read_code({"path": "nope/nada.py"}))

    def test_find_code(self) -> None:
        out = self.agent._tool_find_code({"grep": "class DevonAgent"})
        self.assertIn("devon.py", out)

    def test_db_query_select(self) -> None:
        out = self.agent._tool_db_query(
            {"sql": "SELECT name FROM sqlite_master WHERE type='table' AND name = 'devon_memory'"}
        )
        self.assertIn("devon_memory", out)

    def test_db_query_refuses_non_select(self) -> None:
        self.assertIn("refused", self.agent._tool_db_query({"sql": "DELETE FROM messages"}))
        self.assertIn("refused", self.agent._tool_db_query({"sql": "UPDATE kv_store SET value = 1"}))

    def test_db_query_appends_limit(self) -> None:
        out = self.agent._tool_db_query({"sql": "SELECT name FROM sqlite_master"})
        self.assertIn("row(s)", out)

    def test_message_flow_empty(self) -> None:
        self.assertIn("no messages recorded", self.agent._tool_message_flow({}))

    def test_message_flow_after_turn(self) -> None:
        db = self.context.db
        with db.transaction():
            db.execute(
                "INSERT INTO conversations (id, title, agent, channel, created_at, updated_at) "
                "VALUES (?, ?, 'partner', 'local', 0, 0)",
                ("local:console", "console"),
            )
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("devon-m1", "local:console", "user", "hey wren", "you", "", 0.1),
            )
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("devon-m2", "local:console", "assistant", "hey! back at you", "wren", "mock-7b", 0.2),
            )
        out = self.agent._tool_message_flow({"limit": 5})
        self.assertIn("hey wren", out)
        self.assertIn("back at you", out)
        self.assertIn("mock-7b", out)

    def test_git_status(self) -> None:
        out = self.agent._tool_git_status({})
        self.assertIn("branch:", out)
        self.assertIn("commit:", out)

    def test_logs_tail_missing(self) -> None:
        self.assertIn("no log file", self.agent._tool_logs_tail({}))

    def test_mood_without_brain(self) -> None:
        self.assertIn("no mood state", self.agent._tool_mood({}))

    def test_chat_stats_without_gateway(self) -> None:
        out = self.agent._tool_chat_stats({})
        self.assertIn("live games:", out)

    def test_research_needs_query(self) -> None:
        self.assertIn("query", self.agent._tool_research({}))

    def test_mission_needs_goal(self) -> None:
        self.assertIn("goal", self.agent._tool_mission({}))

    def test_run_tests_small_module(self) -> None:
        out = self.agent._tool_run_tests({"pattern": "tests.test_cli_logging", "timeout": 60})
        self.assertIn("PASS", out)
        self.assertIn("OK", out)

    def test_run_tests_bare_package_uses_discovery(self) -> None:
        # The phone's exit-5 bug: `python -m unittest tests` finds ZERO
        # tests. Bare packages/dirs must use discovery; dotted modules must
        # not (discovery would ignore them).
        cmd = DevonAgent._test_cmd("tests")
        self.assertIn("discover", cmd)
        self.assertIn("-s", cmd)
        self.assertIn("tests", cmd)
        cmd = DevonAgent._test_cmd("tests.test_devon")
        self.assertNotIn("discover", cmd)
        self.assertIn("tests.test_devon", cmd)
        cmd = DevonAgent._test_cmd("tests/test_devon.py")
        self.assertNotIn("discover", cmd)
        self.assertIn("tests.test_devon", cmd)

    def test_models_tool_with_real_router(self) -> None:
        # _make_context builds the real router (mock provider).
        ctx2, tmp2 = _make_context()
        try:
            agent2 = DevonAgent(ctx2)
            out = agent2._tool_models({})
            self.assertIn("provider chain:", out)
            self.assertIn("health=", out)
        finally:
            ctx2.close()
            tmp2.cleanup()

    def test_env_tool(self) -> None:
        out = self.agent._tool_env({})
        self.assertIn("provider chain:", out)
        self.assertIn("power mode:", out)
        self.assertIn("features on:", out)

    def test_history_tool_empty_then_filled(self) -> None:
        self.assertIn("no prior", self.agent._tool_history({"task": "x"}))
        self.context.router = DevonRouter(digest="first investigation digest here.")
        self.agent.run("check the message flow", chat_key="local:console")
        out = self.agent._tool_history({"task": "message flow"})
        self.assertIn("first investigation digest", out)

    def test_trace_tool(self) -> None:
        self.assertIn("no chats recorded", self.agent._tool_trace({}))
        db = self.context.db
        with db.transaction():
            db.execute(
                "INSERT INTO conversations (id, title, agent, channel, created_at, updated_at) "
                "VALUES (?, 't', 'partner', 'local', 0, 0)", ("local:console",)
            )
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("tr1", "local:console", "user", "hey", "you", "", 0.1),
            )
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("tr2", "local:console", "assistant", "hey you", "Wren", "mock-7b", 0.2),
            )
        out = self.agent._tool_trace({"chat_key": "local:console"})
        self.assertIn("hey", out)
        self.assertIn("mock-7b", out)

    def test_git_diff_tool(self) -> None:
        # Hermetic: the sandbox checkout is often shallow, so build a tiny
        # two-commit repo instead of relying on ambient history.
        import subprocess

        with tempfile.TemporaryDirectory(prefix="nm-devon-diff-") as d:
            repo = Path(d)
            env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            }

            def sh(*args: str) -> None:
                subprocess.run(args, cwd=repo, check=True, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            sh("git", "init", "-q")
            (repo / "f.txt").write_text("v1\n")
            sh("git", "add", ".")
            sh("git", "commit", "-qm", "one")
            (repo / "f.txt").write_text("v2\n")
            sh("git", "add", ".")
            sh("git", "commit", "-qm", "two")

            agent = DevonAgent(self.context, repo_root=repo)
            out = agent._tool_git_diff({"ref_a": "HEAD~1", "ref_b": "HEAD"})
            self.assertIn("diff --git", out)
            self.assertIn("v2", out)

    def test_watch_tool_starts(self) -> None:
        out = self.agent._tool_watch({"task": "check the mood", "interval_minutes": 5, "passes": 2})
        self.assertIn("watch started", out)


# ── test-run summaries (the phone saw: "FAIL (exit 5)" with no detail) ───────


class DevonTestRunSummaryTest(unittest.TestCase):
    def test_pass_run(self) -> None:
        from nomorals.agents.devon import summarize_test_run

        out = summarize_test_run("tests", 0, "Ran 989 tests in 35.8s\n\nOK\n")
        self.assertIn("PASS", out)
        self.assertIn("OK", out)
        self.assertNotIn("failing:", out)

    def test_fail_run_names_the_failing_tests(self) -> None:
        from nomorals.agents.devon import summarize_test_run

        output = (
            "test_alpha (tests.test_x.Alpha) ... ok\n"
            "test_beta (tests.test_x.Beta) ... FAIL\n"
            "\n"
            "======================================================================\n"
            "FAIL: test_beta (tests.test_x.Beta)\n"
            "----------------------------------------------------------------------\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in test_beta\n'
            "AssertionError\n"
            "\n"
            "----------------------------------------------------------------------\n"
            "Ran 2 tests in 0.1s\n"
            "\n"
            "FAILED (failures=1)\n"
        )
        out = summarize_test_run("tests", 5, output)
        self.assertIn("FAIL (exit 5)", out)
        self.assertIn("FAILED (failures=1)", out)
        self.assertIn("failing:", out)
        self.assertIn("FAIL: test_beta (tests.test_x.Beta)", out)
        # The failing-test line must appear BEFORE the raw tail noise.
        self.assertLess(out.index("FAIL: test_beta"), out.rindex("AssertionError"))

    def test_fail_run_with_no_output(self) -> None:
        from nomorals.agents.devon import summarize_test_run

        out = summarize_test_run("tests", 5, "")
        self.assertIn("FAIL (exit 5)", out)


# ── memory box ───────────────────────────────────────────────────────────────


class DevonMemoryBoxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.context.router = DevonRouter(
            digest="checked the flow — every recent message has an assistant reply row."
        )
        self.agent = DevonAgent(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_run_persists_steps_and_digest(self) -> None:
        result = self.agent.run("check the message flow", chat_key="local:console")
        self.assertEqual(result.planned_by, "heuristic")
        self.assertTrue(result.steps, "no steps ran")
        rows = self.context.db.query("SELECT * FROM devon_memory WHERE run_id = ?", (result.run_id,))
        self.assertTrue(any(r["status"] == "done" for r in rows), "digest row missing")
        self.assertTrue(
            any(r["status"] == "step" and r["tool"] == "message_flow" for r in rows),
            "message_flow step row missing",
        )
        digests = self.agent.recent_digests()
        self.assertEqual(len(digests), 1)
        self.assertIn("checked the flow", digests[0]["digest"])

    def test_prior_memory_carried_into_next_planner_prompt(self) -> None:
        self.agent.run("how is your mood right now", chat_key="local:console")
        second = DevonRouter(digest="second pass: still fine.")
        self.context.router = second
        agent2 = DevonAgent(self.context)
        agent2.run("mood again", chat_key="local:console")
        self.assertTrue(second.planner_prompts, "planner was never called")
        prompt = second.planner_prompts[-1]
        self.assertIn("Prior memory", prompt)
        self.assertIn("how is your mood right now", prompt)

    def test_digest_survives_crash_between_steps(self) -> None:
        # Steps persist as they happen; a 'done' row only appears at the end.
        # Simulate a crash: write steps, then no finish row.
        result = self.agent.run("check the message flow", chat_key="local:console")
        self.context.db.execute("DELETE FROM devon_memory WHERE status = 'done'")
        rows = self.context.db.query(
            "SELECT tool FROM devon_memory WHERE run_id = ? AND status = 'step'",
            (result.run_id,),
        )
        self.assertTrue(rows, "step rows must survive the 'crash'")


# ── LLM planner ──────────────────────────────────────────────────────────────


class DevonLlmPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.plan = [
            {"tool": "git_status", "args": {}, "why": "orient"},
            {"tool": "mood", "args": {}, "why": "check state"},
        ]
        self.context.router = DevonRouter(plan=self.plan, digest="oriented: branch and mood both look fine.")
        self.agent = DevonAgent(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_llm_plan_drives_run(self) -> None:
        result = self.agent.run("what's up with the system", chat_key="local:console")
        self.assertEqual(result.planned_by, "llm")
        self.assertEqual([s.tool for s in result.steps], ["git_status", "mood"])
        self.assertIn("oriented:", result.digest)

    def test_unknown_tools_dropped_from_plan(self) -> None:
        self.context.router = DevonRouter(
            plan=[
                {"tool": "nuke_the_fridge", "args": {}},
                {"tool": "git_status", "args": {}},
            ],
            digest="x",
        )
        result = self.agent.run("test", chat_key="c")
        self.assertEqual(result.planned_by, "llm")
        self.assertEqual([s.tool for s in result.steps], ["git_status"])

    def test_bad_json_falls_back_to_heuristic(self) -> None:
        # "(no plan)" is not JSON → heuristic kicks in.
        self.context.router = DevonRouter(plan=None, digest="heuristic digest")
        result = self.agent.run("check the message flow", chat_key="c")
        self.assertEqual(result.planned_by, "heuristic")
        self.assertTrue(result.steps)


# ── runtime wiring: /devon is owner-only ─────────────────────────────────────


class DevonRuntimeWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        self.context, self.tmp = _make_context()
        # The brain binds the router at construction time (see /model's
        # explicit swap), so set a default BEFORE building the runtime.
        self.context.router = FakeRouter()
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.brain = self.runtime.brain
        # Presence is real; seed so any brain reply these tests touch is immediate.
        self.brain.presence_rng = random.Random(34)
        self.chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM, peer="you")

    def _swap_router(self, router) -> None:
        """Swap context.router AND rebuild the brain on it (the responder
        keeps its own reference from construction time)."""
        from nomorals.agents.partner_runtime import PartnerBrain

        self.context.router = router
        self.brain = PartnerBrain(self.context, curator=None)
        self.brain.presence_rng = random.Random(34)
        self.runtime.brain = self.brain

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_owner_devon_replies_with_digest(self) -> None:
        self._swap_router(DevonRouter(
            digest="checked the flow — every recent message has an assistant reply row."
        ))
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(
            chat=self.chat, incoming=True,
            text="/devon check if the brain replied to the last messages", sender="you",
        ))
        self.assertTrue(
            _wait(lambda: any("🔧 devon" in s for s in self.adapter.sent), timeout=30.0),
            f"no devon reply: {self.adapter.sent}",
        )
        reply = next(s for s in self.adapter.sent if "🔧 devon" in s)
        self.assertIn("checked the flow", reply)
        # The investigation journaled itself.
        rows = self.context.db.query("SELECT status FROM devon_memory")
        self.assertTrue(any(r["status"] == "done" for r in rows))

    def test_owner_devon_bare_shows_help(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True, text="/devon", sender="you"))
        self.assertTrue(_wait(lambda: any("devon" in s for s in self.adapter.sent)))
        reply = next(s for s in self.adapter.sent if "devon" in s)
        self.assertIn("give me a task", reply)

    def test_non_owner_devon_is_just_text_to_her(self) -> None:
        self._swap_router(FakeRouter(replies=["huh, a slash? whatever — what's up?"]))
        stranger = ChatRef(platform="local", chat_id="bob123", kind=ChatKind.DM, peer="bob")
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(
            chat=stranger, incoming=True, text="/devon hack the bank", sender="bob",
        ))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1))
        self.assertEqual(self.adapter.sent[0], "huh, a slash? whatever — what's up?")
        # Nothing journaled — devon never ran.
        self.assertEqual(self.context.db.query("SELECT id FROM devon_memory"), [])


# ── games in every chat ──────────────────────────────────────────────────────


class GamesAnywhereTest(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.partner_runtime import PartnerRuntime

        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter(replies=["hi all!"] * 10)
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.brain = self.runtime.brain
        self.brain.presence_rng = random.Random(34)
        self.group = ChatRef(platform="local", chat_id="grp5", kind=ChatKind.GROUP, title="mountain folks")
        self.owner = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM, peer="you")

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_member_starts_game_in_a_group(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.group, incoming=True, text="/game rps", sender="bob"))
        self.assertTrue(
            _wait(lambda: any("rps" in s.lower() and "🎮" in s for s in self.adapter.sent)),
            f"game did not start: {self.adapter.sent}",
        )
        row = self.context.db.query_one(
            "SELECT * FROM game_sessions WHERE chat_key = ? AND status = 'active'",
            (self.group.key,),
        )
        self.assertIsNotNone(row, "game session was not persisted for the group")

    def test_member_plain_move_routed_to_live_game(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.group, incoming=True, text="/game rps", sender="bob"))
        self.assertTrue(_wait(lambda: any("rps" in s.lower() for s in self.adapter.sent)))
        self.adapter.sent.clear()
        # A *different* member plays a plain move — no slash needed.
        self.runtime.on_message(ChatMessage(chat=self.group, incoming=True, text="rock", sender="alice"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "move was not routed")
        self.assertNotIn("hi all!", self.adapter.sent[0], "persona answered instead of the game")

    def test_member_list_games_without_args(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.group, incoming=True, text="/game", sender="bob"))
        self.assertTrue(
            _wait(lambda: any("games — start one" in s for s in self.adapter.sent)),
            f"game list not shown: {self.adapter.sent}",
        )

    def test_other_slash_commands_stay_owner_only(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        # Owner's /status still works from the console.
        self.runtime.on_message(ChatMessage(chat=self.owner, incoming=True, text="/status", sender="you"))
        self.assertTrue(_wait(lambda: any("mood:" in s for s in self.adapter.sent)))
        self.adapter.sent.clear()
        # A member's /status in the group is NOT a command: without a mention
        # she stays silent — that silence is the proof it fell through as text.
        self.runtime.on_message(ChatMessage(chat=self.group, incoming=True, text="/status", sender="bob"))
        time.sleep(0.4)
        self.assertEqual(self.adapter.sent, [], "member slash was dispatched as a command")

    def test_owner_still_plays_games_too(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.owner, incoming=True, text="/game trivia", sender="you"))
        self.assertTrue(
            _wait(lambda: any("trivia" in s.lower() and "🎮" in s for s in self.adapter.sent)),
            f"owner game did not start: {self.adapter.sent}",
        )


if __name__ == "__main__":
    unittest.main()
