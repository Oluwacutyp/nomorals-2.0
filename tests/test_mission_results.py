"""Pass #59 — mission RESULTS surface: stored step outputs viewable in chat.

Pins the contract that a completed mission's deliverable is reachable:

- ``/mission results [id|name]`` renders each step's stored output text
  (progress.py ``render_results_text`` / ``payload_text``).
- Honest empty states: no stored outputs is a plain statement, never an
  error and never fabricated content; a live mission's results are
  labeled partial with the completed-step count.
- ``MissionStore.detail`` reports outputs availability additively and
  the status card points at the results only for terminal missions
  with stored outputs.
- The terminal milestone push carries a digest of the final step's
  output + the results pointer, and its exactly-once marking still
  dedupes a second delivery.
- ``nm missions --results <id>`` renders the same content as chat.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.notifier import Notifier  # noqa: F401  (parity w/ f2)
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.cmdline.commands.missions import _cmd_missions
from nomorals.missions import (
    MissionMilestones,
    MissionStatus,
    MissionStore,
    payload_text,
    render_status_text,
)
from nomorals.storage.db import Database


# ── fakes (mirrors tests/test_mission_control.py / test_missions_f2.py) ─────

class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok


class FakeGateway:
    """Records every send; owner channels report running."""

    def __init__(self, live_platforms=("telegram",)):
        self.live = set(live_platforms)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=0,   # 0 == 0 -> quiet hours disabled (deterministic tests)
        quiet_end=0,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test=None, partner=None, gateway=None):
    tmp = tempfile.mkdtemp(prefix="mission-results-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp,
                               partner=partner or make_partner())
    ctx = SimpleNamespace(db=db, settings=settings,
                          extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


PLAN = ("gather sources", "synthesize findings", "write report")


def make_mission(store, name="research — money", goal="Research money",
                 outputs=None, status=None, completed=None, heartbeat=False):
    mission = store.create_new(goal, name=name)
    mission.state["plan"] = [
        {"name": n, "goal": n, "role": "execution",
         "kind": "io", "depends_on": []}
        for n in PLAN
    ]
    if completed is not None:
        mission.state["completed_steps"] = list(completed)
    if outputs is not None:
        mission.state["outputs"] = dict(outputs)
    if heartbeat:
        mission.state["heartbeat"] = {"pid": os.getpid(), "at": time.time()}
    if status is not None:
        mission.status = status
    store.save(mission)
    return mission


class _RuntimeStub:
    """Drives _control_mission without building a full PartnerRuntime."""

    def __init__(self, ctx):
        self.context = ctx
        self._control_mission = PartnerRuntime._control_mission.__get__(self)
        self._find_mission = PartnerRuntime._find_mission
        self._mission_template_spec = PartnerRuntime._mission_template_spec
        self._release_db_thread = PartnerRuntime._release_db_thread.__get__(self)


# ── payload extraction ───────────────────────────────────────────────────────

class PayloadTextTests(unittest.TestCase):
    def test_prefers_text_key(self):
        self.assertEqual(payload_text({"text": "the findings",
                                       "path": "/tmp/x"}), "the findings")

    def test_findings_list_joins(self):
        out = payload_text({"findings": ["one", "two"]})
        self.assertEqual(out, "one\ntwo")

    def test_no_text_key_renders_compact_lines(self):
        out = payload_text({"path": "/tmp/report.md", "bytes": 42})
        self.assertIn("path: /tmp/report.md", out)
        self.assertIn("bytes: 42", out)
        self.assertNotIn("{", out)

    def test_plain_string_and_none(self):
        self.assertEqual(payload_text("raw"), "raw")
        self.assertEqual(payload_text(None), "")
        self.assertEqual(payload_text({}), "")


# ── chat verb ────────────────────────────────────────────────────────────────

class ResultsVerbTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)
        self.rt = _RuntimeStub(self.ctx)

    def call(self, tail, chat_key="telegram:9"):
        return self.rt._control_mission(tail, chat_key=chat_key)

    def test_completed_mission_renders_each_step(self):
        m = make_mission(
            self.store, status=MissionStatus.DONE,
            completed=PLAN,
            outputs={
                "gather sources": {"text": "found 3 credible sources"},
                "synthesize findings": {"text": "money findings body"},
                "write report": {"text": "the final report body"},
            })
        out = self.call(f"results {m.id}")
        self.assertIn("results: research — money [done]", out)
        for step, text in (("gather sources", "found 3 credible sources"),
                           ("synthesize findings", "money findings body"),
                           ("write report", "the final report body")):
            self.assertIn(f"── {step}", out)
            self.assertIn(text, out)
        # plan order, not insertion/alphabetical order
        self.assertLess(out.index("found 3 credible sources"),
                        out.index("money findings body"))
        self.assertLess(out.index("money findings body"),
                        out.index("the final report body"))

    def test_results_by_name_and_empty_ref(self):
        m = make_mission(
            self.store, status=MissionStatus.DONE, completed=PLAN,
            outputs={"write report": {"text": "only the report"}})
        by_name = self.call("results money")
        self.assertIn("only the report", by_name)
        # empty ref mirrors /mission status: default mission resolution
        by_default = self.call("results")
        self.assertIn("only the report", by_default)
        self.assertIn(m.name, by_default)

    def test_terminal_without_outputs_is_honest(self):
        m = make_mission(self.store, status=MissionStatus.DONE,
                         completed=PLAN, outputs=None)
        out = self.call(f"results {m.id}")
        self.assertIn("no results stored", out)
        self.assertIn("without storing step outputs", out)

    def test_running_mission_partial_label(self):
        m = make_mission(
            self.store, status=MissionStatus.RUNNING, heartbeat=True,
            completed=("gather sources",),
            outputs={"gather sources": {"text": "early sources"}})
        out = self.call(f"results {m.id}")
        self.assertIn("partial results — 1 of 3 steps", out)
        self.assertIn("early sources", out)
        self.assertNotIn("synthesize findings", out.split("──", 1)[-1])

    def test_unknown_ref_matches_sibling_style(self):
        out = self.call("results nope-nothing")
        self.assertIn("no mission matching 'nope-nothing'", out)

    def test_long_output_truncated_with_total(self):
        big = "x" * 5000
        m = make_mission(
            self.store, status=MissionStatus.DONE, completed=PLAN,
            outputs={"write report": {"text": big}})
        out = self.call(f"results {m.id}")
        self.assertIn("…truncated — 5000 chars total", out)
        self.assertLess(len(out), 5000)

    def test_usage_mentions_results(self):
        out = self.call("frobnicate")
        self.assertIn("/mission results [id|name]", out)


# ── detail + status pointer ──────────────────────────────────────────────────

class DetailPointerTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)

    def test_detail_reports_outputs_availability(self):
        m = make_mission(
            self.store, status=MissionStatus.DONE, completed=PLAN,
            outputs={"write report": {"text": "r"},
                     "gather sources": {"text": "g"}})
        detail = self.store.detail(m.id)
        self.assertTrue(detail["has_outputs"])
        # plan order even though stored in reverse order
        self.assertEqual(detail["output_steps"],
                         ["gather sources", "write report"])

    def test_detail_no_outputs(self):
        m = make_mission(self.store, status=MissionStatus.DONE)
        detail = self.store.detail(m.id)
        self.assertFalse(detail["has_outputs"])
        self.assertEqual(detail["output_steps"], [])

    def test_status_pointer_only_terminal_with_outputs(self):
        m = make_mission(
            self.store, status=MissionStatus.DONE, completed=PLAN,
            outputs={"write report": {"text": "the report"}})
        text = render_status_text(self.store.detail(m.id))
        self.assertIn(f"results: /mission results {m.id}", text)

        bare = make_mission(self.store, name="bare",
                            status=MissionStatus.DONE, completed=PLAN)
        text = render_status_text(self.store.detail(bare.id))
        self.assertNotIn("/mission results", text)

        live = make_mission(
            self.store, name="live", status=MissionStatus.RUNNING,
            heartbeat=True, completed=("gather sources",),
            outputs={"gather sources": {"text": "g"}})
        text = render_status_text(self.store.detail(live.id))
        self.assertNotIn("/mission results", text)


# ── terminal milestone push ──────────────────────────────────────────────────

class TerminalDigestTests(unittest.TestCase):
    def setUp(self):
        self.gateway = FakeGateway()
        self.ctx, _ = make_ctx(self, gateway=self.gateway)
        self.store = MissionStore(self.ctx.db)

    def test_terminal_push_carries_digest_and_pointer_once(self):
        m = make_mission(
            self.store, status=MissionStatus.DONE, completed=PLAN,
            outputs={
                "gather sources": {"text": "sources body"},
                "write report": {"text": "FINAL REPORT BODY"},
            })
        reporter = MissionMilestones(self.ctx, store=self.store)
        res = reporter.on_terminal(m, MissionStatus.DONE)
        self.assertTrue(res.get("delivered"))
        self.assertEqual(len(self.gateway.sent), 1)
        text = self.gateway.sent[0][2]
        self.assertIn("FINAL REPORT BODY", text)
        self.assertIn(f"full results: /mission results {m.id}", text)
        # exactly-once marking still dedupes a catch-up re-delivery
        again = reporter.on_terminal(m, MissionStatus.DONE)
        self.assertEqual(again.get("suppressed"), "already-notified")
        self.assertEqual(len(self.gateway.sent), 1)

    def test_terminal_push_without_outputs_unchanged(self):
        m = make_mission(self.store, status=MissionStatus.DONE,
                         completed=PLAN)
        reporter = MissionMilestones(self.ctx, store=self.store)
        res = reporter.on_terminal(m, MissionStatus.DONE)
        self.assertTrue(res.get("delivered"))
        text = self.gateway.sent[0][2]
        self.assertNotIn("/mission results", text)


# ── CLI parity ───────────────────────────────────────────────────────────────

class CliResultsTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)

    def _args(self, **over):
        base = dict(start="", resume="", resume_all=False, pause="",
                    cancel="", resume_status="", replan="", show="",
                    results="", status="", json=False, no_reflect=False,
                    max_iterations=8, budget_wall=0.0, budget_tokens=0,
                    accept="", require_artifact=[])
        base.update(over)
        return argparse.Namespace(**base)

    def test_cli_results_renders_same_content(self):
        m = make_mission(
            self.store, status=MissionStatus.DONE, completed=PLAN,
            outputs={"write report": {"text": "cli-visible report"}})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _cmd_missions(self._args(results=m.id), self.ctx)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("── write report", out)
        self.assertIn("cli-visible report", out)

    def test_cli_results_unknown_id_is_usage_error(self):
        buf = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(err):
            rc = _cmd_missions(self._args(results="nope-id"), self.ctx)
        self.assertEqual(rc, 2)
        self.assertIn("missions:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
