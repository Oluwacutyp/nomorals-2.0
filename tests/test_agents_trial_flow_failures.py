"""Tail-85 pins for nomorals/agents/trial/flow.py — 5 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was taken (trigger == fallback AND control !=
fallback, or the control proving the fallback EQUALS the module's own
designed value). The file itself contains no exception handlers, so it
adds zero census sites, and no sys.path manipulation — tests bind via
rootdir from the tree root.

NOT-recording is pinned with the snapshot method: flow.py records
diligently elsewhere via agents.failures.record_failure, so both the
agency and agents ledgers must be unchanged across each trigger —
none of these five sites records.

Verdict notes (full grounds in SURVEY.md):
- 147: gateway.status() failure -> [] from active_delivery_platforms,
  byte-identical to the genuine no-platforms-running value (and to
  the gateway-is-None value). The only consumer (_delivery_targets)
  still appends the originating `via` platform, and deliver() falls
  back to showing the credential inline with an honest "no live
  channel" audit line — never a false "sent" claim.
- 796 / 1401 / 1635: release_thread_connection cleanup inside
  finally blocks -> pass. Teardown: in every case the guarded work
  (assist-run close-out, resumed-watch close, watch close) completes
  and reports honestly with the release broken; the poll/task
  outcome is never swallowed by these guards (the enclosing except
  paths are separate, unguarded-by-these-sites handlers).
- 1843: sess.close() in run_browser_task's finally -> pass. The task
  result returns and the task's own exception propagates unchanged
  past the broken close.
"""

import json
import os
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents.trial import flow as flow_mod
from nomorals.agents.trial.flow import TrialFlow
from nomorals.storage.db import Database


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _flow_harness():
    tmp = tempfile.mkdtemp()
    db = Database(":memory:")
    db.execute(
        "CREATE TABLE IF NOT EXISTS kv_store ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
        "expires_at REAL)")
    ctx = SimpleNamespace(settings=SimpleNamespace(home=tmp), db=db)
    return TrialFlow(ctx), db, tmp


def _boom_release(db_):
    raise RuntimeError("release boom")


def test_147_status_failure_is_no_platforms():
    before = _snap()

    class BoomGateway:
        def status(self):
            raise RuntimeError("gateway boom")

    assert flow_mod.active_delivery_platforms(BoomGateway()) == []

    class Gw:
        def status(self):
            return {"telegram": {"running_in_session": True},
                    "whatsapp": {"running_in_session": False},
                    "local": {"running_in_session": True}}

    # control: local excluded by design, stopped excluded, live kept
    assert flow_mod.active_delivery_platforms(Gw()) == ["telegram"]
    # [] is also the genuine no-gateway value
    assert flow_mod.active_delivery_platforms(None) == []
    assert _snap() == before


def test_796_assist_run_closeout_survives_broken_release():
    before = _snap()
    flow, _db, _tmp = _flow_harness()
    notes = []
    flow._notify_owner = lambda *a, **k: notes.append((a, k))
    for rid in ("r1", "r2"):
        flow._assist_runs[rid] = {
            "platform": "github", "chat_key": "tg:1",
            "started": time.time(), "state": "queued", "note": ""}
    with mock.patch.object(TrialFlow, "_personal_vault",
                           side_effect=RuntimeError("vault boom")):
        with mock.patch(
                "nomorals.storage.db.release_thread_connection",
                _boom_release):
            out = flow._assist_run("r1", "github", "tg:1", {})
        # control: working release — the identical honest outcome
        out2 = flow._assist_run("r2", "github", "tg:1", {})
    assert out is None and out2 is None
    assert flow._assist_runs["r1"]["state"] == "failed"
    assert "failed" in flow._assist_runs["r1"]["note"]
    assert flow._assist_runs["r2"]["state"] == "failed"
    assert len(notes) == 2
    assert _snap() == before


def test_1401_resumed_watch_closes_survives_broken_release(tmp_path):
    before = _snap()
    # file-backed + migrated: the resumed poll runs on a worker
    # thread whose per-thread connection must see the same DB, and
    # the recovery Notifier's durable store needs its tables —
    # exactly as in production, where the bot DB is always migrated
    db = Database(str(tmp_path / "t.db"))
    db.migrate()
    db.executescript(flow_mod._SMS_WATCHES_DDL)
    now = time.time()
    info = {"number": "+1555000111", "masked": "+1•••0111"}
    db.execute(
        "INSERT INTO trial_sms_watches (watch_id, number, number_info,"
        " chat_key, started, deadline, timeout, state, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        ("w1", "+1555000111", json.dumps(info), "tg:1", now - 5,
         now + 120, 300.0, "watching", now - 5))
    ctx = SimpleNamespace(settings=SimpleNamespace(home=str(tmp_path)),
                          db=db, gateway=None)
    with mock.patch("nomorals.accounts.temp_sms.wait_code",
                    return_value="654321"):
        with mock.patch(
                "nomorals.storage.db.release_thread_connection",
                _boom_release):
            recovered = TrialFlow.recover_interrupted_watches(ctx)
            assert recovered and recovered[0]["watch_id"] == "w1"
            # poll INSIDE the patch window: the resumed thread
            # imports wait_code when it runs, not when it starts
            row = None
            for _ in range(200):
                row = db.query_one(
                    "SELECT state, code FROM trial_sms_watches"
                    " WHERE watch_id=?", ("w1",))
                if row and row["state"] == "done":
                    break
                time.sleep(0.05)
    assert row is not None and row["state"] == "done"
    assert row["code"] == "654321"
    assert _snap() == before


def test_1635_sms_watch_closeout_survives_broken_release():
    before = _snap()
    flow, _db, _tmp = _flow_harness()
    notes = []
    flow._notify_owner = lambda *a, **k: notes.append((a, k))
    flow._sms_watches["w9"] = {"number": "+1555000222",
                               "started": time.time(), "state": "watching"}
    with mock.patch("nomorals.accounts.temp_sms.wait_code",
                    return_value="111222"):
        with mock.patch(
                "nomorals.storage.db.release_thread_connection",
                _boom_release):
            out = flow._run_sms_watch(
                "w9", {"number": "+1555000222"}, "tg:1", "+1555000222",
                30.0)
    assert out is None
    assert flow._sms_watches["w9"]["state"] == "done"
    assert any("111222" in str(a) for a, _k in notes)
    # control: no code arrives — the timeout close also completes
    flow._sms_watches["w10"] = {"number": "+1555000333",
                                "started": time.time(),
                                "state": "watching"}
    with mock.patch("nomorals.accounts.temp_sms.wait_code",
                    return_value=""):
        with mock.patch(
                "nomorals.storage.db.release_thread_connection",
                _boom_release):
            out = flow._run_sms_watch(
                "w10", {"number": "+1555000333"}, "tg:1", "+1555000333",
                30.0)
    assert out is None
    assert flow._sms_watches["w10"]["state"] == "timeout"
    assert _snap() == before


def test_1843_browser_task_close_is_best_effort():
    before = _snap()

    class FakeSess:
        def __init__(self, fail_task=False):
            self.fail_task = fail_task
            self.closed = False

        def task(self, steps=None):
            if self.fail_task:
                raise ValueError("task boom")
            return {"ok": True, "steps": len(steps or [])}

        def close(self):
            self.closed = True
            raise RuntimeError("close boom")

    sess = FakeSess()
    with mock.patch("nomorals.tools.browser.get_rendered_session",
                    return_value=sess):
        out = TrialFlow.run_browser_task([{"act": "open"}])
    assert out == {"ok": True, "steps": 1}
    assert sess.closed is True
    # control: the task's own exception propagates past the broken
    # close — teardown never masks the real failure
    sess2 = FakeSess(fail_task=True)
    with mock.patch("nomorals.tools.browser.get_rendered_session",
                    return_value=sess2):
        with pytest.raises(ValueError, match="task boom"):
            TrialFlow.run_browser_task([{"act": "open"}])
    assert sess2.closed is True
    assert _snap() == before
