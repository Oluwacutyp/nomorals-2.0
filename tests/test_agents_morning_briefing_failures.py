"""Pass tail-9 pins: agents/morning_briefing failures degrade honestly.

Target: nomorals/agents/morning_briefing.py — 0 converts / 10 exclusions.
Every site below maps a failure to a genuine degraded value (the
documented miss/omit value of that function) and the briefing proceeds
honestly — no section claims what it does not have, no outcome is
fabricated, nothing is silently dropped that a consumer depends on.
None of them records, because none of them hides a user-visible outcome:
the module's own failures ledger (morning_briefing_failures) is reused
read-only for the NOT-recording pins.

NOT-recording method (tail-8 improvement): each test snapshots
recent_failures() immediately before its trigger and asserts it
unchanged after — pollution-proof against other tests in the same
session, strictly stronger than asserting empty.

No bare ``except: pass`` anywhere in this file — the scanner counts
test files too; the tests below need no except blocks at all (stubs
raise, the code under test catches; sys.modules patches are restored
in finally blocks).
"""

import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

from nomorals.agents import morning_briefing as mb
from nomorals.agents.morning_briefing_failures import recent_failures as _rf


def _snap():
    return list(_rf())


def _assert_no_record(before):
    assert list(_rf()) == before


def _patch_module(name, **attrs):
    orig = sys.modules.get(name)
    stub = ModuleType(name)
    for k, v in attrs.items():
        setattr(stub, k, v)
    sys.modules[name] = stub
    return orig


def _unpatch(name, orig):
    if orig is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = orig


def test_238_optional_market_provider_import_failure_is_none():
    # Module-level optional import: a build without the integration
    # gets the documented None miss value, not a crash. The tree root
    # is derived from this file's location so the test is cwd-proof.
    import os
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from types import ModuleType;"
        "m = ModuleType('nomorals.integrations.sentinel_bridge');"
        "sys.modules['nomorals.integrations.sentinel_bridge'] = m;"
        "from nomorals.agents import morning_briefing as mb2;"
        "assert mb2.MarketDataProvider is None"
    ) % tree
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=tree)
    assert r.returncode == 0, r.stderr


def test_334_watcher_store_import_failure_is_none():
    orig = _patch_module("nomorals.agents.watchers")
    try:
        before = _snap()
        assert mb._watcher_store(SimpleNamespace(db=None)) is None
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.agents.watchers", orig)


def test_428_calendar_tool_failures_fall_through_to_empty():
    class BoomTools:
        def call(self, *a, **k):
            raise RuntimeError("tools boom")

    before = _snap()
    out = mb.CalendarProvider()._today(
        SimpleNamespace(tools=BoomTools(), calendar=None))
    assert out == []
    _assert_no_record(before)


def test_473_scorer_import_failure_keeps_nudges_working():
    orig = _patch_module("nomorals.memory.delivery")
    try:
        before = _snap()
        assert mb.PeopleProvider()._scorer is None
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.memory.delivery", orig)


def test_479_cadence_import_failure_omits_people_section():
    orig = _patch_module("nomorals.agents.proactive")
    try:
        before = _snap()
        assert mb.PeopleProvider().collect(SimpleNamespace(), 0.0) is None
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.agents.proactive", orig)


def test_687_rooms_import_failure_omits_rooms_section():
    orig = _patch_module("nomorals.workspace.rooms")
    try:
        before = _snap()
        ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir="/tmp"),
                              db=None)
        assert mb.RoomsProvider().collect(ctx, 0.0) is None
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.workspace.rooms", orig)


def test_731_skill_edits_query_failure_skips_only_that_line():
    class BoomDb:
        def query_one(self, *a, **k):
            raise RuntimeError("skill_edits boom")

    class Sched:
        def __init__(self, ctx):
            pass

        def list_jobs(self):
            return [{"id": "1", "name": "x", "last_status": "failed",
                     "last_run_at": time.time(), "last_error": "boom"}]

    orig_w = _patch_module("nomorals.agents.watchers")
    orig_s = _patch_module("nomorals.agents.scheduler", Scheduler=Sched)
    try:
        before = _snap()
        sec = mb.DevonSelfProvider().collect(SimpleNamespace(db=BoomDb()), 0.0)
        assert sec is not None
        assert not any("🔧" in ln for ln in sec.lines)
        assert any("job failed" in ln for ln in sec.lines)
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.agents.watchers", orig_w)
        _unpatch("nomorals.agents.scheduler", orig_s)


def test_792_missing_upgrade_queue_module_omits_research_section():
    orig = _patch_module("nomorals.agents.upgrade_queue")
    try:
        before = _snap()
        assert mb.ResearchProvider()._collect(SimpleNamespace(), 0.0) is None
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.agents.upgrade_queue", orig)


def test_800_upgrade_queue_list_failure_omits_research_section():
    class FakeQ:
        def __init__(self, ctx=None):
            pass

        def list(self, **kw):
            raise RuntimeError("queue list boom")

    orig = _patch_module("nomorals.agents.upgrade_queue", UpgradeQueue=FakeQ)
    try:
        before = _snap()
        assert mb.ResearchProvider()._collect(SimpleNamespace(), 0.0) is None
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.agents.upgrade_queue", orig)


def test_1553_scheduler_list_failure_treated_as_unscheduled():
    class Sched:
        calls = []

        def __init__(self, ctx):
            pass

        def list_jobs(self):
            raise RuntimeError("scheduler table boom")

        def add(self, *a, **k):
            Sched.calls.append((a, k))
            return {"id": "new"}

    orig = _patch_module("nomorals.agents.scheduler", Scheduler=Sched)
    try:
        before = _snap()
        ctx = SimpleNamespace(settings=SimpleNamespace(
            briefing=SimpleNamespace(time="07:00"), workspace_dir="/tmp"))
        out = mb.ensure_briefing_job(ctx)
        assert len(Sched.calls) == 1
        assert out.get("name") == "briefing"
        assert "already_scheduled" not in out
        _assert_no_record(before)
    finally:
        _unpatch("nomorals.agents.scheduler", orig)
