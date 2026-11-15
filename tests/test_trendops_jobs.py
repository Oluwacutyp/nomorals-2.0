"""TrendOps unprompted-proposal hook (media/trendops/jobs.py) — tests.

The owner vision (2026-10-21): the class PROPOSES applications
unprompted.  What is pinned here:

* the scheduler job registers with the exact spec + payload, exactly
  once, and is NOT routine — a routine job's successes stay silent
  and the proposal would never reach the owner;
* the ``trendops`` tool binding binds against the real registry and
  its ``propose`` action returns the ranked proposals;
* the job runs end to end through the real scheduler: a live format
  produces a published proposal (the notifier receives the report
  Doc carrying the ready apply command), and when no formats are
  live the run is silent (scheduled silence — journaled, not posted);
* the report builder names the most urgent format in its summary
  and renders every proposal's why + apply command; an empty
  proposals list is classified "nothing to report", never posted.

The TrendOps library root is patched to ``tmp_path`` in every test
that touches the tool: the tool builds ``TrendOps(context)`` with
the module default root, and the default root is the real home —
never touched by a test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.trendops.jobs import (
    TRENDOPS_PROPOSALS_JOB,
    ensure_trendops_jobs,
)


class _SchedNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body,
                           "doc": kw.get("doc")})
        return {"delivered": True, "channel": "test"}


def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(
        resolve=lambda rel: tmp_path / rel,
        home_path=tmp_path, workspace_dir=str(tmp_path))
    return SimpleNamespace(db=db, settings=settings, extras={},
                           tools=None)


@pytest.fixture
def _tmp_library(monkeypatch, tmp_path):
    """Point the TrendOps default root at tmp_path (never the home)."""
    import nomorals.media.trendops.editor as editor_mod

    monkeypatch.setattr(editor_mod, "DEFAULT_ROOT",
                        tmp_path / "trendops")
    return tmp_path / "trendops"


def _expire_all_formats(root: Path) -> None:
    """Expire every format on disk so propose() returns [].

    The library re-seeds the roadster only when the formats dir is
    empty, so expiring the seed (rather than removing it) is what
    yields a genuinely empty live set.
    """
    from nomorals.media.trendops.library import FormatLibrary

    lib = FormatLibrary(root)
    for slug in lib.list(live_only=False):
        fmt = lib.get(slug, live_only=False)
        fmt.expires_at = time.time() - 1.0
        lib.save(fmt)


def test_job_registers_exactly_once_with_exact_payload(tmp_path):
    from nomorals.agents.scheduler import Scheduler

    ctx = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = _SchedNotifier()
    out = ensure_trendops_jobs(sched)
    assert [o["name"] for o in out] == [TRENDOPS_PROPOSALS_JOB]
    rows = [j for j in sched.list_jobs()
            if j["name"] == TRENDOPS_PROPOSALS_JOB]
    assert len(rows) == 1
    # list_jobs presents a daily job as kind + HH:MM detail (the
    # "daily" prefix is the spec syntax, not stored — cf. cron jobs,
    # which re-add their prefix in _format_job).
    assert rows[0]["kind"] == "daily"
    assert rows[0]["spec"] == "09:00"
    # NOT routine: the proposal IS the delivery — a routine job's
    # successes stay silent and the owner would never see it.
    assert rows[0]["routine"] is False
    row = ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        (TRENDOPS_PROPOSALS_JOB,))
    assert json.loads(row["payload"]) == {
        "tool": "trendops", "args": {"action": "propose"}}
    # Idempotent: a second registration changes nothing.
    ensure_trendops_jobs(sched)
    assert len([j for j in sched.list_jobs()
                if j["name"] == TRENDOPS_PROPOSALS_JOB]) == 1


def test_tool_registered_and_binding_binds(tmp_path, _tmp_library):
    import inspect

    from nomorals.tools.registry import ToolRegistry

    ctx = _sched_ctx(tmp_path)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    spec = registry._tools.get("trendops")
    assert spec is not None, "trendops tool not registered"
    # The scheduler's exact call shape binds against the real signature.
    inspect.signature(spec.fn).bind(action="propose")
    outcome = registry.call("trendops", action="propose")
    assert outcome.ok, getattr(outcome.error, "message", outcome.error)
    body = outcome.unwrap()
    assert isinstance(body.get("proposals"), list)
    # The fresh tmp library seeds the roadster, live for 30 days.
    assert len(body["proposals"]) == 1
    assert body["proposals"][0]["slug"] == "roadster"
    assert body["proposals"][0]["apply"].startswith(
        "/trendops apply roadster ")


def test_job_runs_end_to_end_delivers_proposal(
        tmp_path, _tmp_library):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.social.chat.present import render_text
    from nomorals.tools.registry import ToolRegistry

    ctx = _sched_ctx(tmp_path)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    sched = Scheduler(ctx)
    notifier = _SchedNotifier()
    sched.notifier = notifier
    ensure_trendops_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == TRENDOPS_PROPOSALS_JOB][0]
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    assert out["silent"] is False, out
    stored = ctx.db.query_one(
        "SELECT last_result FROM schedule_jobs WHERE id = ?",
        (row["id"],))
    assert not (stored["last_result"] or "").startswith("job failed"), \
        stored["last_result"]
    # The proposal reached the owner: exactly one publish, carrying
    # the report Doc with the ready apply command.
    assert len(notifier.calls) == 1
    call = notifier.calls[0]
    assert call["kind"] == "schedule"
    assert call["doc"] is not None
    text = render_text(call["doc"], "telegram-bot")
    assert "roadster" in text
    assert "/trendops apply roadster" in text


def test_job_silent_when_no_live_formats(tmp_path, _tmp_library):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.tools.registry import ToolRegistry

    _expire_all_formats(_tmp_library)
    ctx = _sched_ctx(tmp_path)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    sched = Scheduler(ctx)
    notifier = _SchedNotifier()
    sched.notifier = notifier
    ensure_trendops_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == TRENDOPS_PROPOSALS_JOB][0]
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    # Scheduled silence: the run executed and journaled, nothing posted.
    assert out["silent"] is True, out
    assert notifier.calls == []


def test_report_builder_renders_proposals():
    from nomorals.agents.job_reports import job_report_doc, noop_reason
    from nomorals.social.chat.present import render_text

    value = {"proposals": [
        {"slug": "roadster", "name": "Roadster GTA", "niche": "gta",
         "days_left": 2.5, "uses": 0,
         "apply": "/trendops apply roadster <footage>",
         "why": "Roadster GTA peaks now — 2d left, used 0x"},
        {"slug": "hormozi-cut", "name": "Hormozi Cut",
         "niche": "business", "days_left": 9.0, "uses": 3,
         "apply": "/trendops apply hormozi-cut <footage>",
         "why": "Hormozi Cut peaks now — 9d left, used 3x"},
    ]}
    assert noop_reason("trendops", value) is None
    doc = job_report_doc("trendops", value,
                         job_name=TRENDOPS_PROPOSALS_JOB)
    assert doc.summary == "2 formats peaking — Roadster GTA, 2.5d left."
    # The Doc's own lines carry the raw apply commands; the rendered
    # form HTML-escapes the <footage> placeholder on Telegram.
    lines = [line for section in doc.sections
             for line in (section.quote or ())]
    assert any("/trendops apply roadster <footage>" in line
               for line in lines)
    assert any("/trendops apply hormozi-cut <footage>" in line
               for line in lines)
    text = render_text(doc, "telegram-bot")
    assert "/trendops apply roadster" in text
    assert "/trendops apply hormozi-cut" in text
    assert "Roadster GTA peaks now" in text
    # Empty proposals: nothing to report, never posted.
    assert noop_reason("trendops", {"proposals": []}) == \
        "nothing to report"
    empty = job_report_doc("trendops", {"proposals": []},
                           job_name=TRENDOPS_PROPOSALS_JOB)
    assert empty.summary == "No live trend formats."
