"""Single-registration contract for memory consolidation.

History: consolidation was registered TWICE in PartnerRuntime.start() —
the organ loop added a "memory consolidation" job (every 12h, tool
``memory_consolidate``, hardcoded) and, ~300 lines later, the cadence
registrar added "memory-consolidation" (every
``consolidation_interval_seconds``, tool ``memory`` action
``consolidate``).  Both fired the same additive tick: consolidation
double-ran, the 12h copy ignored the configured interval (including
disable), and its runs bypassed the cadence bookkeeping (last run /
run count / status).

The cadence job is canonical — it is the one named by status(),
health(), /memory schedule and the adaptive-buttons surface, the one
whose interval is configurable, and the one whose tick records itself.
This suite pins:

* ensure registers exactly one consolidation job, named
  ``memory-consolidation`` with the configured spec/payload;
* a second registration attempt is a no-op (registrar reports
  already-scheduled; the scheduler's own exact-replay dedupe agrees);
* a persisted legacy duplicate row is removed by the registrar —
  including when the canonical job already exists (the live-DB state);
* the surviving job still fires its handler through the scheduler.
"""

from types import SimpleNamespace

from nomorals.agents.scheduler import Scheduler
from nomorals.memory import cadence
from nomorals.storage.db import Database


class _Tools:
    """Recording tool registry stub (the Pass #71 harness shape)."""

    def __init__(self):
        self.calls = []

    def call(self, tool, capabilities=None, **args):
        self.calls.append((tool, dict(args)))
        return SimpleNamespace(
            ok=True, value={"ok": True, "summaries": 0}, error=None)


class _Notifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


def _ctx():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, tools=_Tools(),
                           settings=SimpleNamespace(),
                           memory=SimpleNamespace(), extras={})


def _sched(ctx):
    sched = Scheduler(ctx)
    sched.notifier = _Notifier()
    return sched


def _consolidation_jobs(sched):
    names = {cadence.CONSOLIDATION_JOB_NAME,
             cadence.LEGACY_CONSOLIDATION_JOB_NAME}
    return [j for j in sched.list_jobs() if j.get("name") in names]


def _add_legacy_job(sched):
    """The removed organ-loop registration, exactly as it was added."""
    return sched.add(cadence.LEGACY_CONSOLIDATION_JOB_NAME, "every 12h",
                     "tool", {"tool": "memory_consolidate", "args": {}})


def test_ensure_registers_exactly_one_consolidation_job():
    ctx = _ctx()
    out = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    assert out["scheduled"] is True
    jobs = _consolidation_jobs(_sched(ctx))
    assert len(jobs) == 1
    job = jobs[0]
    assert job["name"] == cadence.CONSOLIDATION_JOB_NAME
    assert job["spec"] == "every 3600s"
    assert job["payload_kind"] == "tool"


def test_second_registration_attempt_is_a_noop():
    ctx = _ctx()
    first = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    assert first["scheduled"] is True
    second = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    assert second.get("already_scheduled") is True
    assert second["job_id"] == first["job_id"]
    assert len(_consolidation_jobs(_sched(ctx))) == 1
    # the scheduler's own contract agrees: an exact replay of the same
    # add dedupes instead of minting a second row
    sched = _sched(ctx)
    replay = sched.add(cadence.CONSOLIDATION_JOB_NAME, "every 3600s",
                       "tool",
                       {"tool": "memory",
                        "args": {"action": "consolidate"}})
    assert replay.get("deduped") is True
    assert replay["id"] == first["job_id"]
    assert len(_consolidation_jobs(sched)) == 1


def test_legacy_duplicate_removed_when_canonical_already_exists():
    # the live-DB state this fix exists for: BOTH rows persisted, the
    # canonical one already correct — ensure must still purge the legacy
    # row on the already-scheduled path, not return early past it
    ctx = _ctx()
    sched = _sched(ctx)
    _add_legacy_job(sched)
    first = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    assert first["scheduled"] is True
    assert first.get("legacy_removed") == 1
    jobs = _consolidation_jobs(_sched(ctx))
    assert [j["name"] for j in jobs] == [cadence.CONSOLIDATION_JOB_NAME]
    second = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    assert second.get("already_scheduled") is True
    assert not second.get("legacy_removed")
    assert len(_consolidation_jobs(_sched(ctx))) == 1


def test_legacy_duplicate_removed_when_it_is_the_only_row():
    ctx = _ctx()
    sched = _sched(ctx)
    _add_legacy_job(sched)
    out = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    assert out["scheduled"] is True
    assert out.get("legacy_removed") == 1
    jobs = _consolidation_jobs(_sched(ctx))
    assert [j["name"] for j in jobs] == [cadence.CONSOLIDATION_JOB_NAME]


def test_surviving_job_fires_the_consolidation_handler():
    ctx = _ctx()
    out = cadence.ensure_consolidation_job(
        ctx, scheduler_factory=Scheduler)
    sched = _sched(ctx)
    result = sched.run_now(out["job_id"])
    assert result["ok"] is True
    # exactly one handler call, and it is the cadence tick: the memory
    # tool's consolidate action (persona.register -> consolidate_now),
    # never the legacy memory_consolidate direct call
    assert ctx.tools.calls == [("memory", {"action": "consolidate"})]
