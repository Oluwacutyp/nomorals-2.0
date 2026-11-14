"""Journal trace/replay view: reconstruction, replay, HTML, CLI, registry.

The fixture journal is one conversation with two turns: two user
messages, two assistant replies, three tool calls (one ok, one error,
one denied — plus one outside the time window that must never appear),
one audit row, one cognition row, one skill use, and a second newer
conversation for the list view.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.storage.db import Database
from nomorals.tools import trace_view
from nomorals.tools.trace_view import (
    TraceBuilder,
    TraceNotFound,
    replay_trace,
    set_model_runner,
    set_tool_runner,
    trace_page,
    write_trace_page,
)

CONV = "conv-1"
CONV2 = "conv-2"

M2_TEXT = "Here is your plan: wake at 6, work 7-12."
M4_TEXT = "Done — gym at 6pm."


def _fixture_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    db.insert("conversations", {
        "id": CONV, "title": "help me plan", "agent": "devon",
        "channel": "telegram", "created_at": 1000.0, "updated_at": 1100.0})
    db.insert("conversations", {
        "id": CONV2, "title": "second", "agent": "devon",
        "channel": "cli", "created_at": 1150.0, "updated_at": 1200.0})
    for row in (
        {"id": "m1", "conversation_id": CONV, "role": "user",
         "content": "plan my day", "created_at": 1001.0},
        {"id": "m2", "conversation_id": CONV, "role": "assistant",
         "content": M2_TEXT, "model": "codebeast-3.8b", "tokens": 42,
         "created_at": 1005.0},
        {"id": "m3", "conversation_id": CONV, "role": "user",
         "content": "and add gym", "created_at": 1010.0},
        {"id": "m4", "conversation_id": CONV, "role": "assistant",
         "content": M4_TEXT, "model": "codebeast-3.8b", "tokens": 30,
         "created_at": 1015.0},
        {"id": "m5", "conversation_id": CONV2, "role": "user",
         "content": "hello", "created_at": 1151.0},
    ):
        db.insert("messages", row)
    for row in (
        {"id": "t1", "actor": "devon", "tool": "search",
         "capability": "net.out", "decision": "allow", "args_digest": "aabbcc",
         "status": "ok", "duration_ms": 120.0, "created_at": 1002.0},
        {"id": "t2", "actor": "devon", "tool": "read_file",
         "capability": "fs.read", "decision": "allow", "args_digest": "ddeeff",
         "status": "error", "duration_ms": 5.0, "error": "no such file",
         "created_at": 1003.0},
        {"id": "t3", "actor": "devon", "tool": "delete_file",
         "capability": "fs.delete", "decision": "deny", "args_digest": "001122",
         "status": "denied", "duration_ms": 0.0, "created_at": 1004.0},
        # Outside the conversation's time window: must never appear.
        {"id": "t4", "actor": "devon", "tool": "search",
         "capability": "net.out", "decision": "allow", "args_digest": "334455",
         "status": "ok", "duration_ms": 50.0, "created_at": 2000.0},
    ):
        db.insert("tool_calls", row)
    db.insert("audit_log", {
        "id": "a1", "ts": 1006.0, "actor": "devon", "action": "power.unlock",
        "capability": "sys.config", "decision": "allow",
        "detail": json.dumps({"x": 1})})
    db.insert("cognition_log", {
        "id": "c1", "ts": 1007.0,
        "stages": json.dumps({"perceive": {}, "decide": {}}),
        "seconds": 0.5, "note": "quick"})
    db.insert("skill_uses", {
        "id": "s1", "skill_id": "planner", "task": "plan", "outcome": "done",
        "used_at": 1008.0, "success": 1, "context": "",
        "created_at": 1008.0})
    return db


def _row_counts(db: Database) -> dict[str, int]:
    out = {}
    for table in ("conversations", "messages", "tool_calls", "audit_log",
                  "cognition_log", "skill_uses"):
        row = db.query_one(f"SELECT COUNT(*) AS n FROM {table}")
        out[table] = int((row or {}).get("n") or 0)
    return out


class TraceBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _fixture_db()
        self.builder = TraceBuilder(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_list_newest_first_with_counts(self) -> None:
        rows = self.builder.list_traces()
        self.assertEqual([r["conversation_id"] for r in rows], [CONV2, CONV])
        self.assertEqual(rows[1]["messages"], 4)
        self.assertEqual(rows[0]["messages"], 1)

    def test_list_channel_filter(self) -> None:
        rows = self.builder.list_traces(channel="telegram")
        self.assertEqual([r["conversation_id"] for r in rows], [CONV])

    def test_spans_in_time_order_and_window(self) -> None:
        trace = self.builder.build(CONV)
        kinds = [s.kind for s in trace.spans]
        self.assertEqual(kinds, ["message", "tool", "tool", "tool", "message",
                                 "audit", "cognition", "skill", "message",
                                 "message"])
        ids = [s.id for s in trace.spans]
        self.assertNotIn("t4", ids)  # outside the window
        ts = [s.ts for s in trace.spans]
        self.assertEqual(ts, sorted(ts))

    def test_join_marks(self) -> None:
        trace = self.builder.build(CONV)
        by_kind = {}
        for span in trace.spans:
            by_kind.setdefault(span.kind, set()).add(span.join)
        self.assertEqual(by_kind["message"], {"conversation"})
        self.assertEqual(by_kind["tool"], {"time-window"})
        self.assertEqual(by_kind["audit"], {"time-window"})

    def test_missing_fields_are_named_never_invented(self) -> None:
        trace = self.builder.build(CONV)
        by_id = {s.id: s for s in trace.spans}
        self.assertEqual(by_id["m2"].missing, ["latency_ms", "cost"])
        self.assertEqual(by_id["t1"].missing, ["args", "result"])
        self.assertEqual(by_id["a1"].missing, [])
        self.assertEqual(by_id["c1"].missing, [])
        self.assertEqual(by_id["s1"].missing, [])
        # The tool span carries what the journal does record.
        self.assertEqual(by_id["t1"].detail["args_digest"], "aabbcc")
        self.assertEqual(by_id["t2"].detail["error"], "no such file")
        self.assertEqual(by_id["m2"].detail["model"], "codebeast-3.8b")
        self.assertEqual(by_id["m2"].detail["tokens"], 42)

    def test_turns_split_on_user_messages(self) -> None:
        trace = self.builder.build(CONV)
        self.assertEqual(len(trace.turns), 2)
        self.assertEqual(trace.turns[0].user_text, "plan my day")
        self.assertEqual(trace.turns[1].user_text, "and add gym")
        all_ids = [sid for t in trace.turns for sid in t.span_ids]
        self.assertEqual(sorted(all_ids), sorted(s.id for s in trace.spans))
        self.assertEqual(trace.turns[0].span_ids[0], "m1")
        self.assertEqual(trace.turns[1].span_ids[0], "m3")

    def test_stats(self) -> None:
        trace = self.builder.build(CONV)
        self.assertEqual(trace.stats["spans"], 10)
        self.assertEqual(trace.stats["by_kind"], {
            "message": 4, "tool": 3, "audit": 1, "cognition": 1, "skill": 1})
        self.assertEqual(trace.stats["tool_status"], {
            "ok": 1, "error": 1, "denied": 1})
        self.assertEqual(trace.stats["total_tool_ms"], 125.0)
        self.assertEqual(trace.stats["total_tokens"], 72)
        self.assertEqual(trace.stats["turns"], 2)

    def test_unknown_conversation(self) -> None:
        with self.assertRaises(TraceNotFound):
            self.builder.build("no-such-conversation")

    def test_to_dict_truncates_long_content(self) -> None:
        self.db.insert("messages", {
            "id": "m-long", "conversation_id": CONV, "role": "user",
            "content": "x" * 5000, "created_at": 1020.0})
        trace = self.builder.build(CONV)
        data = trace.to_dict(truncate=100)
        long_span = [s for s in data["spans"] if s["id"] == "m-long"][0]
        self.assertTrue(long_span["input_truncated"])
        self.assertLessEqual(len(long_span["input"]), 101)


class TraceReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _fixture_db()
        self.trace = TraceBuilder(self.db).build(CONV)
        self.before = _row_counts(self.db)

    def tearDown(self) -> None:
        set_model_runner(None)
        set_tool_runner(None)
        self.db.close()

    def _recorded_runner(self, prompt):
        # Returns the recorded reply for the prompt's length.
        if len(prompt) == 1:
            return M2_TEXT
        return M4_TEXT

    def test_model_same_and_user_skipped(self) -> None:
        result = replay_trace(self.trace, model_runner=self._recorded_runner)
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "same")
        self.assertEqual(by_id["m4"]["verdict"], "same")
        self.assertEqual(by_id["m1"]["verdict"], "skipped")
        self.assertEqual(by_id["m1"]["reason"], "input, not re-run")
        self.assertEqual(result.verdicts["same"], 2)

    def test_model_changed(self) -> None:
        result = replay_trace(self.trace,
                              model_runner=lambda prompt: "something else")
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "changed")
        self.assertEqual(by_id["m2"]["replayed"], "something else")
        self.assertEqual(by_id["m2"]["recorded"], M2_TEXT)

    def test_model_error_diverges(self) -> None:
        def boom(prompt):
            raise RuntimeError("provider down")

        result = replay_trace(self.trace, model_runner=boom)
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "diverged")
        self.assertIn("provider down", by_id["m2"]["reason"])

    def test_model_empty_diverges(self) -> None:
        result = replay_trace(self.trace,
                              model_runner=lambda prompt: "   ")
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "diverged")

    def test_no_runner_skips_model_spans(self) -> None:
        result = replay_trace(self.trace)
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "skipped")
        self.assertEqual(by_id["m2"]["reason"], "no model runner attached")

    def test_tool_spans_need_allow_tools(self) -> None:
        result = replay_trace(self.trace, model_runner=self._recorded_runner)
        by_id = {r["span_id"]: r for r in result.results}
        for tid in ("t1", "t2", "t3"):
            self.assertEqual(by_id[tid]["verdict"], "skipped")
            self.assertIn("--allow-tools", by_id[tid]["reason"])

    def test_tool_spans_need_args(self) -> None:
        result = replay_trace(self.trace, model_runner=self._recorded_runner,
                              allow_tools=True)
        by_id = {r["span_id"]: r for r in result.results}
        for tid in ("t1", "t2", "t3"):
            self.assertEqual(by_id[tid]["verdict"], "skipped")
            self.assertIn("digest only", by_id[tid]["reason"])

    def test_tool_write_capability_never_executes(self) -> None:
        calls: list[tuple[str, dict]] = []

        def runner(tool, args):
            calls.append((tool, args))
            return {"ok": True}

        result = replay_trace(
            self.trace, model_runner=self._recorded_runner,
            tool_runner=runner, allow_tools=True,
            tool_args={"t1": {"q": "x"}, "t3": {"path": "/tmp/x"}})
        by_id = {r["span_id"]: r for r in result.results}
        # net.out and fs.delete are writes: skipped, runner never called.
        self.assertEqual(by_id["t1"]["verdict"], "skipped")
        self.assertIn("would write", by_id["t1"]["reason"])
        self.assertIn("net.out", by_id["t1"]["reason"])
        self.assertEqual(by_id["t3"]["verdict"], "skipped")
        self.assertIn("fs.delete", by_id["t3"]["reason"])
        self.assertEqual(calls, [])

    def test_tool_read_only_replays_with_no_baseline(self) -> None:
        def runner(tool, args):
            return {"content": "file contents here"}

        result = replay_trace(
            self.trace, model_runner=self._recorded_runner,
            tool_runner=runner, allow_tools=True,
            tool_args={"t2": {"path": "notes.txt"}})
        by_id = {r["span_id"]: r for r in result.results}
        # fs.read + args supplied + allowed: runs, but the journal holds
        # no recorded output to compare against.
        self.assertEqual(by_id["t2"]["verdict"], "replayed")
        self.assertIn("file contents here", by_id["t2"]["replayed"])

    def test_observation_spans_skipped(self) -> None:
        result = replay_trace(self.trace, model_runner=self._recorded_runner,
                              allow_tools=True)
        by_id = {r["span_id"]: r for r in result.results}
        for sid in ("a1", "c1", "s1"):
            self.assertEqual(by_id[sid]["verdict"], "skipped")
            self.assertEqual(by_id[sid]["reason"],
                             "observation only — nothing to re-run")

    def test_timeout_diverges(self) -> None:
        def slow(prompt):
            time.sleep(0.5)
            return "late"

        result = replay_trace(self.trace, model_runner=slow,
                              span_timeout_s=0.05)
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "diverged")
        self.assertIn("timed out", by_id["m2"]["reason"])

    def test_max_spans_truncates(self) -> None:
        result = replay_trace(self.trace, model_runner=self._recorded_runner,
                              max_spans=2)
        self.assertTrue(result.truncated)
        self.assertEqual(result.spans_considered, 2)
        self.assertEqual(result.spans_total, 10)
        self.assertEqual(len(result.results), 2)

    def test_replay_writes_nothing(self) -> None:
        replay_trace(self.trace, model_runner=self._recorded_runner,
                     allow_tools=True,
                     tool_runner=lambda t, a: {"ok": True},
                     tool_args={"t2": {"path": "x"}})
        self.assertEqual(_row_counts(self.db), self.before)

    def test_seam_runners_used_when_args_omitted(self) -> None:
        set_model_runner(self._recorded_runner)
        result = replay_trace(self.trace)
        by_id = {r["span_id"]: r for r in result.results}
        self.assertEqual(by_id["m2"]["verdict"], "same")


class TracePageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _fixture_db()
        self.trace = TraceBuilder(self.db).build(CONV)

    def tearDown(self) -> None:
        self.db.close()

    def test_page_renders_turns_and_spans(self) -> None:
        page = trace_page(self.trace)
        self.assertIn("trace: help me plan", page)
        self.assertIn("turn 1", page)
        self.assertIn("turn 2", page)
        self.assertIn("tool search", page)
        self.assertIn("plan my day", page)
        self.assertIn("Content-Security-Policy", page)
        self.assertIn("time-window join", page)
        self.assertIn("missing from the journal", page)

    def test_page_escapes_injected_markup(self) -> None:
        self.db.insert("messages", {
            "id": "m-evil", "conversation_id": CONV, "role": "user",
            "content": "</script><script>alert(1)</script>",
            "created_at": 1020.0})
        page = trace_page(TraceBuilder(self.db).build(CONV))
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;/script&gt;", page)

    def test_write_page_and_bad_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_trace_page(trace_page(self.trace), "conv-1",
                                    out_dir=tmp)
            self.assertEqual(path, Path(tmp) / "conv-1.html")
            self.assertTrue(path.is_file())
            self.assertIn("trace", path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            write_trace_page("<html/>", "../escape", out_dir="/tmp")


class TraceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _fixture_db()
        from nomorals.tools.registry import ToolRegistry

        self.reg = ToolRegistry()
        self.reg.context = SimpleNamespace(db=self.db)
        trace_view.register(self.reg)

    def tearDown(self) -> None:
        self.db.close()

    def test_trace_list_tool(self) -> None:
        out = self.reg.call("trace_list", limit=1)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.value["count"], 1)
        self.assertEqual(out.value["traces"][0]["conversation_id"], CONV2)

    def test_trace_show_tool(self) -> None:
        out = self.reg.call("trace_show", conversation_id=CONV)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.value["conversation_id"], CONV)
        self.assertEqual(len(out.value["spans"]), 10)

    def test_trace_show_unknown(self) -> None:
        out = self.reg.call("trace_show", conversation_id="nope")
        self.assertFalse(out.ok)

    def test_trace_replay_tool_no_runner(self) -> None:
        out = self.reg.call("trace_replay", conversation_id=CONV)
        self.assertTrue(out.ok, out.error)
        by_id = {r["span_id"]: r for r in out.value["results"]}
        self.assertEqual(by_id["m2"]["verdict"], "skipped")

    def test_no_database_attached(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = SimpleNamespace(db=None)
        trace_view.register(reg)
        out = reg.call("trace_list")
        self.assertFalse(out.ok)


class TraceCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "cli.db")
        db = Database(self.db_path)
        db.migrate()
        # Same fixture, file-backed so the CLI can open it.
        db.insert("conversations", {
            "id": CONV, "title": "help me plan", "agent": "devon",
            "channel": "telegram", "created_at": 1000.0,
            "updated_at": 1100.0})
        db.insert("messages", {
            "id": "m1", "conversation_id": CONV, "role": "user",
            "content": "plan my day", "created_at": 1001.0})
        db.insert("messages", {
            "id": "m2", "conversation_id": CONV, "role": "assistant",
            "content": M2_TEXT, "model": "codebeast-3.8b", "tokens": 42,
            "created_at": 1005.0})
        db.close()
        from nomorals.cmdline.commands.trace import _cmd_trace

        self._cmd_trace = _cmd_trace
        self.context = SimpleNamespace(
            settings=SimpleNamespace(db_path=self.db_path))

    def tearDown(self) -> None:
        set_model_runner(None)
        self.tmp.cleanup()

    def _args(self, action: str, **kw: Any) -> SimpleNamespace:
        base = {"trace_action": action, "json": False, "limit": 10,
                "channel": "", "agent": "", "conversation_id": "",
                "html": None, "max_spans": 50, "timeout": 30.0,
                "allow_tools": False, "tool_args": ""}
        base.update(kw)
        return SimpleNamespace(**base)

    def _run(self, args: SimpleNamespace) -> tuple[int, str]:
        buf = StringIO()
        with redirect_stdout(buf):
            rc = self._cmd_trace(args, self.context)
        return rc, buf.getvalue()

    def test_cli_list(self) -> None:
        rc, out = self._run(self._args("list"))
        self.assertEqual(rc, 0)
        self.assertIn(CONV, out)
        self.assertIn("2 messages", out)

    def test_cli_list_json(self) -> None:
        rc, out = self._run(self._args("list", json=True))
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["traces"][0]["conversation_id"], CONV)

    def test_cli_show(self) -> None:
        rc, out = self._run(self._args("show", conversation_id=CONV))
        self.assertEqual(rc, 0)
        self.assertIn("turn 1", out)
        self.assertIn("plan my day", out)
        self.assertIn(M2_TEXT, out)

    def test_cli_show_unknown(self) -> None:
        rc, out = self._run(self._args("show", conversation_id="nope"))
        self.assertEqual(rc, 1)

    def test_cli_show_html(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            rc, out = self._run(self._args(
                "show", conversation_id=CONV, html=out_dir))
            self.assertEqual(rc, 0)
            self.assertIn("trace page:", out)
            self.assertTrue((Path(out_dir) / "conv-1.html").is_file())

    def test_cli_replay_with_fake_runner(self) -> None:
        set_model_runner(lambda prompt: M2_TEXT)
        rc, out = self._run(self._args("replay", conversation_id=CONV))
        self.assertEqual(rc, 0)
        self.assertIn("same: 1", out)

    def test_cli_replay_bad_tool_args(self) -> None:
        rc, out = self._run(self._args(
            "replay", conversation_id=CONV, tool_args="not json"))
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
