"""Small remainders bundle: scheduled weekly review + ``nm decisions`` + ``nm offline``.

Failing-first: on the pristine base the three new surfaces do not exist
(no ``nomorals.reviews.jobs``, no ``nm decisions`` / ``nm offline``
commands, no offline state summary), so every test here fails there.

1. Scheduled weekly review — ``build_weekly_review`` exists (reviews/
   review.py) and ``/focus week`` renders it on demand, but no
   ``ensure_*`` job ever ran it. The new ``reviews/jobs.py`` registers a
   durable weekly job (Sundays 20:00 local — the week is still the one
   ending then; ``week_bounds`` is Monday 00:00 UTC → +7d) whose tick
   builds the review from the real ledgers and publishes it via the
   Notifier, once per week (watermarked), skipping empty weeks.
2. ``nm decisions`` — the decision journal (decisions/journal.py) is
   wired in chat as /decisions only; ``nm notify decisions`` is the
   alert-policy verdicts, a different thing. The new top-level command
   mirrors the chat verbs (list/add/resolve/review/calibration/whatif).
3. ``nm offline`` — the offline fallback is tick + in-chat only. The new
   command surfaces the real durable state: the offline queue (rows
   exist only after a failed send), the offline sync job, the BrainGate
   held-reply episodes from the failure ledger, and the offline matrix.

The backup ``register_jobs()`` migration is deliberately NOT part of
this lane (backup/jobs.py documents it as a non-migration).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

_WED = _dt.datetime(2026, 11, 11, 12, 0, tzinfo=_dt.UTC).timestamp()


def _ctx(home: Path):
    return SimpleNamespace(
        db=None,
        settings=SimpleNamespace(home=str(home), home_path=Path(home)),
    )


def _ns(**kwargs):
    args = argparse.Namespace(
        decisions_action="list", decisions_rest=[], offline_action="status",
        json=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


class _FakeNotifier:
    def __init__(self):
        self.calls: list[dict] = []

    def publish(self, kind, title="", body="", **kwargs):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True}


class _FakeScheduler:
    """Minimal stand-in for the real Scheduler's job surface."""

    def __init__(self):
        self.jobs: list[dict] = []
        self._seq = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, payload_kind, payload, **kwargs):
        self._seq += 1
        job = {"id": f"job-{self._seq}", "name": name,
               "spec": spec, "payload_kind": payload_kind,
               "payload": payload, **kwargs}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j["id"] != job_id]
        return len(self.jobs) < before


def _seed_week(tmp: Path, now: float = _WED):
    """One kept promise + one resolved decision inside the week."""
    from nomorals.commitments.ledger import CommitmentLedger
    from nomorals.decisions.journal import DecisionJournal

    data = tmp / "data"
    data.mkdir(parents=True, exist_ok=True)
    ledger = CommitmentLedger(str(data / "commitments.json"))
    c = ledger.add(what="Ship the onboarding flow",
                   deadline=now + 86400, ts=now - 86400)
    ledger.mark_kept(c.id, "shipped it", ts=now - 3600)
    journal = DecisionJournal(str(data / "decisions.json"))
    d = journal.record(decision="Ship the onboarding flow",
                       chosen="ship", prediction="users love it",
                       confidence=0.8, ts=now - 86400)
    journal.resolve(d.id, outcome="shipped, no regressions", success=True)
    return c, d


class WeeklyReviewJobsTests(unittest.TestCase):
    def test_ensure_registers_weekly_job_and_is_idempotent(self):
        from nomorals.reviews.jobs import (
            WEEKLY_REVIEW_JOB,
            WEEKLY_REVIEW_SPEC,
            WEEKLY_REVIEW_TOOL,
            ensure_weekly_review_job,
        )

        sched = _FakeScheduler()
        first = ensure_weekly_review_job(sched)
        self.assertEqual(first["name"], WEEKLY_REVIEW_JOB)
        self.assertEqual(first["spec"], WEEKLY_REVIEW_SPEC)
        self.assertEqual(first["payload_kind"], "tool")
        self.assertEqual(first["payload"]["tool"], WEEKLY_REVIEW_TOOL)
        self.assertTrue(first.get("routine"))
        second = ensure_weekly_review_job(sched)
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(len(sched.jobs), 1)

    def test_ensure_replaces_stale_spec(self):
        from nomorals.reviews.jobs import (
            WEEKLY_REVIEW_JOB,
            ensure_weekly_review_job,
        )

        sched = _FakeScheduler()
        stale = sched.add(WEEKLY_REVIEW_JOB, "every 24h", "tool",
                          {"tool": "__weekly_review_tick__"}, routine=True)
        out = ensure_weekly_review_job(sched)
        self.assertNotEqual(out["id"], stale["id"])
        self.assertEqual(len(sched.jobs), 1)

    def test_tick_publishes_review_from_real_ledgers(self):
        from nomorals.reviews.jobs import run_weekly_review_tick_once

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_week(tmp)
            notifier = _FakeNotifier()
            out = run_weekly_review_tick_once(
                context=_ctx(tmp), notifier=notifier, now=_WED)
            self.assertTrue(out["ok"], out)
            self.assertEqual(len(out["posted"]), 1)
            self.assertEqual(len(notifier.calls), 1)
            body = notifier.calls[0]["body"]
            self.assertIn("Ship the onboarding flow", body)
            self.assertIn("Kept promise", body)
            self.assertEqual(notifier.calls[0]["kind"], "weekly_review")

    def test_tick_skips_empty_week_and_writes_no_watermark(self):
        from nomorals.reviews.jobs import run_weekly_review_tick_once

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            notifier = _FakeNotifier()
            out = run_weekly_review_tick_once(
                context=_ctx(tmp), notifier=notifier, now=_WED)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["posted"], [])
            self.assertTrue(out["skipped"])
            self.assertEqual(notifier.calls, [])
            self.assertFalse((tmp / "data" / "weekly_review.json").exists())

    def test_tick_second_run_same_week_skips(self):
        from nomorals.reviews.jobs import run_weekly_review_tick_once

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_week(tmp)
            notifier = _FakeNotifier()
            first = run_weekly_review_tick_once(
                context=_ctx(tmp), notifier=notifier, now=_WED)
            self.assertEqual(len(first["posted"]), 1)
            second = run_weekly_review_tick_once(
                context=_ctx(tmp), notifier=notifier, now=_WED + 3600)
            self.assertTrue(second["ok"], second)
            self.assertEqual(second["posted"], [])
            self.assertTrue(second["skipped"])
            self.assertEqual(len(notifier.calls), 1)

    def test_tick_next_week_delivers_again(self):
        from nomorals.reviews.jobs import run_weekly_review_tick_once

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_week(tmp)
            notifier = _FakeNotifier()
            run_weekly_review_tick_once(
                context=_ctx(tmp), notifier=notifier, now=_WED)
            # A kept promise inside the NEXT week re-arms the review.
            from nomorals.commitments.ledger import CommitmentLedger

            ledger = CommitmentLedger(
                str(tmp / "data" / "commitments.json"))
            nxt = _WED + 7 * 86400
            c2 = ledger.add(what="Call the bank", ts=nxt - 86400)
            ledger.mark_kept(c2.id, "called", ts=nxt - 3600)
            out = run_weekly_review_tick_once(
                context=_ctx(tmp), notifier=notifier, now=nxt)
            self.assertTrue(out["ok"], out)
            self.assertEqual(len(out["posted"]), 1)
            self.assertEqual(len(notifier.calls), 2)

    def test_tick_never_raises_without_context(self):
        from nomorals.reviews.jobs import run_weekly_review_tick_once

        out = run_weekly_review_tick_once(context=None)
        self.assertFalse(out["ok"])
        self.assertTrue(out["error"])

    def test_tick_tool_registered_in_autonomy_registry(self):
        from nomorals.tools import autonomy as autonomy_tools

        class FakeRegistry:
            def __init__(self, context):
                self.context = context
                self.fns = {}

            def register(self, name, fn=None, **kw):
                if fn is None:
                    def deco(f):
                        self.fns[name] = f
                        return f
                    return deco
                self.fns[name] = fn
                return fn

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_week(tmp)
            reg = FakeRegistry(_ctx(tmp))
            autonomy_tools.register(reg)
            self.assertIn("__weekly_review_tick__", reg.fns)
            # Calling through the registry never raises (the tick is
            # scheduler-internal: a failure returns ok False).
            out = reg.fns["__weekly_review_tick__"]()
            self.assertIsInstance(out, dict)
            self.assertIn("ok", out)


class DecisionsCliTests(unittest.TestCase):
    def _run(self, tmp: Path, **kwargs):
        from nomorals.cmdline.commands.decisions import _cmd_decisions

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = _cmd_decisions(_ns(**kwargs), _ctx(tmp))
        return rc, out_buf.getvalue() + err_buf.getvalue()

    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(Path(td))
            self.assertEqual(rc, 0)
            self.assertIn("no open decisions", out)

    def test_add_list_resolve_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rc, out = self._run(
                tmp, decisions_action="add",
                decisions_rest=["Ship the onboarding flow", "|", "ship",
                                "|", "users love it", "|", "80%"])
            self.assertEqual(rc, 0, out)
            self.assertIn("recorded:", out)
            did = re.search(r"recorded: (\S+):", out).group(1)
            rc, out = self._run(tmp)
            self.assertIn("Ship the onboarding flow", out)
            self.assertIn(did, out)
            rc, out = self._run(
                tmp, decisions_action="resolve",
                decisions_rest=[did, "good", "shipped, no regressions"])
            self.assertEqual(rc, 0, out)
            self.assertIn("resolved:", out)
            self.assertIn("shipped, no regressions", out)
            self.assertIn("🧭", out)
            rc, out = self._run(tmp)
            self.assertIn("no open decisions", out)

    def test_add_bad_confidence_is_usage_error(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(
                Path(td), decisions_action="add",
                decisions_rest=["Pick a lane", "|", "a", "|", "b", "|", "abc"])
            self.assertEqual(rc, 2)
            self.assertIn("not a number", out)

    def test_review_and_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._run(tmp, decisions_action="add",
                      decisions_rest=["Ship it", "|", "yes", "|", "good",
                                      "|", "70%"])
            rc, out = self._run(tmp, decisions_action="calibration")
            self.assertEqual(rc, 0)
            self.assertIn("no resolved decisions yet", out)
            rc, out = self._run(tmp, decisions_action="review",
                                decisions_rest=["weekly"])
            self.assertEqual(rc, 0)
            self.assertIn("Ship it", out)

    def test_whatif_renders(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_week(tmp)
            rc, out = self._run(
                tmp, decisions_action="add",
                decisions_rest=["Ship the onboarding flow", "|", "ship",
                                "|", "users love it", "|", "80%",
                                "|", "Ship the onboarding flow; Wait a week"])
            self.assertEqual(rc, 0, out)
            self.assertIn("what-if for", out)

    def test_unknown_action_is_usage_error(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(Path(td), decisions_action="explode")
            self.assertEqual(rc, 2)

    def test_parser_and_dispatch_cover_decisions(self):
        from nomorals.cmdline import _COMMANDS, _parser

        parsed = _parser().parse_args(["decisions", "list"])
        self.assertEqual(parsed.decisions_action, "list")
        self.assertIn("decisions", _COMMANDS)
        self.assertTrue(callable(_COMMANDS["decisions"].handler))


class OfflineCliTests(unittest.TestCase):
    def test_status_empty_queue(self):
        from nomorals.agents.offline_host import summarize_offline_state
        from nomorals.offline import OfflineQueue

        with tempfile.TemporaryDirectory() as td:
            queue = OfflineQueue(root=Path(td) / "offline")
            out = summarize_offline_state(queue=queue, scheduler=None,
                                          failure_entries=[])
            self.assertEqual(out["queue"]["pending"], 0)
            self.assertEqual(out["queue"]["failed"], 0)
            self.assertEqual(out["queue"]["total"], 0)
            self.assertFalse(out["sync_job"]["scheduled"])
            self.assertEqual(out["braingate"]["episodes"], 0)
            self.assertEqual(out["braingate"]["open"], 0)
            self.assertTrue(out["matrix"])
            self.assertIn("email_send", [row["feature"] for row in out["matrix"]])

    def test_status_queued_actions(self):
        from nomorals.agents.offline_host import summarize_offline_state
        from nomorals.offline import OfflineQueue

        with tempfile.TemporaryDirectory() as td:
            queue = OfflineQueue(root=Path(td) / "offline")
            a1 = queue.enqueue("email_send", {"to": "a@b.c"},
                               entity="draft-1")
            a2 = queue.enqueue("email_send", {"to": "d@e.f"},
                               entity="draft-2")
            a2.status = "failed"
            a2.error = "connection reset"
            queue.update(a2)
            out = summarize_offline_state(queue=queue, scheduler=None,
                                          failure_entries=[], now=_WED)
            self.assertEqual(out["queue"]["pending"], 1)
            self.assertEqual(out["queue"]["failed"], 1)
            self.assertEqual(out["queue"]["total"], 2)
            self.assertEqual(out["queue"]["by_kind"], {"email_send": 2})
            self.assertEqual(out["queue"]["oldest_pending_id"], a1.id)
            self.assertIn("draft-2", json.dumps(out["queue"]))

    def test_status_braingate_episodes(self):
        from nomorals.agents.offline_host import summarize_offline_state
        from nomorals.offline import OfflineQueue

        entries = [
            {"ts": 1000.0,
             "site": "runtime.PartnerRuntime._send_reply:braingate",
             "exc_type": "BrainGateHeld", "message": "personal replies held"},
            {"ts": 2000.0,
             "site": "runtime.PartnerRuntime._send_reply:braingate",
             "exc_type": "BrainGateRecovered",
             "message": "brain recovered — a model rung answered again; "
                        "personal replies resumed (3 held during the "
                        "episode opened on telegram, 600s)"},
            {"ts": 3000.0,
             "site": "runtime.PartnerRuntime._send_reply:braingate",
             "exc_type": "BrainGateHeld", "message": "personal replies held"},
            {"ts": 4000.0, "site": "somewhere.else",
             "exc_type": "ValueError", "message": "unrelated"},
        ]
        with tempfile.TemporaryDirectory() as td:
            queue = OfflineQueue(root=Path(td) / "offline")
            out = summarize_offline_state(queue=queue, scheduler=None,
                                          failure_entries=entries)
            self.assertEqual(out["braingate"]["episodes"], 2)
            self.assertEqual(out["braingate"]["open"], 1)
            self.assertEqual(out["braingate"]["held_closed"], 3)
            self.assertTrue(out["braingate"]["held_open_unknown"])

    def test_cli_status_and_queue(self):
        from nomorals.cmdline.commands.offline import _cmd_offline
        from nomorals.offline import OfflineQueue

        with tempfile.TemporaryDirectory() as td:
            queue = OfflineQueue(root=Path(td) / "offline")
            queue.enqueue("email_send", {"to": "a@b.c"}, entity="draft-9")
            ctx = _ctx(Path(td))
            ctx.offline_queue = queue
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_offline(_ns(offline_action="status"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("1 pending", buf.getvalue())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_offline(_ns(offline_action="queue"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("draft-9", buf.getvalue())

    def test_cli_unknown_action_is_usage_error(self):
        from nomorals.cmdline.commands.offline import _cmd_offline

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_offline(_ns(offline_action="explode"), _ctx(Path(td)))
            self.assertEqual(rc, 2)

    def test_parser_and_dispatch_cover_offline(self):
        from nomorals.cmdline import _COMMANDS, _parser

        parsed = _parser().parse_args(["offline", "status"])
        self.assertEqual(parsed.offline_action, "status")
        self.assertIn("offline", _COMMANDS)
        self.assertTrue(callable(_COMMANDS["offline"].handler))


if __name__ == "__main__":
    unittest.main()
