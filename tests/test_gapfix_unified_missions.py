"""Gapfix lane 69 — the unified scheduler actually carries live work.

Before the fix (audit gap R1-11 remainder, register section 8):

* ``UnifiedScheduler.register_organ_tick`` has ZERO production
  callers — the R1-11 ship (``c3e4140b``) built the 5m driver, but no
  existing live organ tick ever moved onto it.
* No mission producer exists: nothing in production ever calls
  ``schedule_mission`` on the unified scheduler.
* No mission handler is registered at boot: nothing in production
  ever calls ``register_handler`` on the boot-built stack.

These tests drive the PRODUCTION entry points (the new
``nomorals.autonomy.missions`` module, the ``remove_wal_checkpoint_job``
retirement in ``storage.db``, and the ``__unified_scheduler_tick__``
tool) — never the dormant pieces in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace

NOW = 1780000000.0


class _Clock:
    """Controllable clock: the stack, the ticks and the producer all
    read the same "now" through it."""

    def __init__(self, now: float = NOW) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class _FakeDb:
    """Workspace-DB stand-in: only ``checkpoint_maintenance`` matters."""

    def __init__(self) -> None:
        self.calls = 0

    def checkpoint_maintenance(self) -> dict:
        self.calls += 1
        return {"busy": False, "log_pages": 0,
                "wal_bytes_before": 0, "wal_bytes_after": 0,
                "reclaimed_bytes": 0}


class _FakeScheduler:
    """Minimal live-Scheduler shape for ensure/remove job tests."""

    def __init__(self, jobs=None):
        self.jobs = list(jobs or [])
        self.add_calls = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, *, routine=None):
        job = {"id": f"job-{len(self.jobs)}", "name": name,
               "spec": spec, "kind": kind, "payload": payload,
               "routine": routine}
        self.jobs.append(job)
        self.add_calls += 1
        return job

    def remove(self, ref):
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j["id"] != ref]
        return len(self.jobs) < before


def _stack(tmp_path, clock):
    from nomorals.autonomy.wiring import build_stack

    return build_stack(tmp_path, clock=clock)


def _ledger_with_due(*, clock, overdue=True, future=False):
    """A ledger with one due commitment (deadline in the past) and,
    optionally, one far-future commitment that must never produce."""
    from nomorals.commitments.ledger import CommitmentLedger

    ledger = CommitmentLedger()
    due = None
    if overdue:
        due = ledger.add(promisee="Ada", what="call back",
                         deadline=clock() - 60, ts=clock())
    far = None
    if future:
        far = ledger.add(promisee="Chidi", what="review the draft",
                         deadline=clock() + 30 * 86400, ts=clock())
    return ledger, due, far


def _notify_sink():
    """A notify callable capturing every (kind, title, body)."""
    calls: list = []

    def _notify(kind, title, body):
        calls.append((kind, title, body))
        return {"ok": True, "delivered": 1}

    return calls, _notify


def _scheduler_ctx(tmp_path, *, home=None):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(
        partner=SimpleNamespace(owner_chats="telegram:111"),
        resolve=lambda rel: tmp_path / rel,
        workspace_dir=str(tmp_path),
        home=str(home if home is not None else tmp_path),
    )
    return SimpleNamespace(db=db, settings=settings, tools=None, extras={})


def _capture_tools(ctx):
    """Register the autonomy tools into a capturing registry."""
    from nomorals.tools import autonomy as autonomy_tools

    captured: dict = {}

    class _Registry:
        def __init__(self):
            self.context = ctx

        def register(self, name, fn=None, **kw):
            if fn is None:
                def _deco(f):
                    captured[name] = f
                    return f
                return _deco
            captured[name] = fn
            return fn

    autonomy_tools.register(_Registry())
    return captured


# ── the old durable job is retired ────────────────────────────────────


def test_remove_wal_checkpoint_job_removes_named_job():
    """The migration retires the old durable job by name: the
    ``__wal_checkpoint__`` tool stops firing from the live scheduler
    the moment the tick moves onto the unified scheduler — otherwise
    the checkpoint would fire twice (old job + unified tick)."""
    from nomorals.storage.db import (
        WAL_CHECKPOINT_JOB,
        remove_wal_checkpoint_job,
    )

    old = {"id": "job-0", "name": WAL_CHECKPOINT_JOB, "spec": "every 1h"}
    other = {"id": "job-1", "name": "something else", "spec": "every 1h"}
    fake = _FakeScheduler([old, other])
    assert remove_wal_checkpoint_job(fake) is True
    assert [j["name"] for j in fake.list_jobs()] == ["something else"]
    assert remove_wal_checkpoint_job(fake) is False  # idempotent
    assert remove_wal_checkpoint_job(None) is False  # no scheduler, no raise


# ── the tick moves onto the unified scheduler ─────────────────────────


def test_register_boot_autonomy_registers_wal_tick(tmp_path):
    """Boot registers the WAL checkpoint as an organ tick on the
    shared stack: name, hourly interval, storage organ — and a second
    boot replaces the registration instead of doubling it."""
    from nomorals.autonomy.missions import register_boot_autonomy

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock, overdue=False)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    tick = stack.scheduler._ticks["wal-checkpoint"]
    assert tick["interval"] == 3600.0
    assert tick["organ"] == "storage"
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    assert len([n for n in stack.scheduler._ticks
                if n == "wal-checkpoint"]) == 1


def test_wal_tick_fires_exactly_once_per_interval(tmp_path):
    """One firing per hour through the unified path: due at boot, then
    silent for the rest of the interval, then exactly one more firing
    at the next interval — never twice in one window."""
    from nomorals.autonomy.missions import register_boot_autonomy

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock, overdue=False)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    assert stack.scheduler.run_due(now=clock())  # first pass fires it
    assert db.calls == 1
    clock.advance(60)
    stack.scheduler.run_due(now=clock())
    assert db.calls == 1  # still inside the interval: silent
    clock.advance(3540)
    stack.scheduler.run_due(now=clock())
    assert db.calls == 2  # next interval: exactly one more firing


def test_wal_tick_exactly_once_across_stacks(tmp_path):
    """Two runtimes share one home and one queue file: the tick's
    durable last-run marker means the second stack does NOT re-fire
    a checkpoint the first stack already ran inside the interval —
    the marker, not memory, is the claim."""
    from nomorals.autonomy.missions import register_boot_autonomy

    clock = _Clock()
    stack_a = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock, overdue=False)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack_a, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    stack_a.scheduler.run_due(now=clock())
    assert db.calls == 1

    stack_b = _stack(tmp_path, clock)  # a second runtime, same home
    register_boot_autonomy(stack_b, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    clock.advance(1800)
    stack_b.scheduler.run_due(now=clock())
    assert db.calls == 1  # marker says A ran it 30m ago: skipped
    # The skip still advances the tick's next_due a full interval from
    # the skip, so the next firing lands an hour after the skip, not
    # on the first stack's boundary — at most once an hour either way.
    clock.advance(3600)
    stack_b.scheduler.run_due(now=clock())
    assert db.calls == 2  # interval elapsed: exactly one firing


def test_register_boot_autonomy_removes_old_wal_job(tmp_path):
    """The boot migration removes the stale durable job from the live
    scheduler in the same breath as it registers the tick — the old
    path is gone, the unified path is the only firing path."""
    from nomorals.autonomy.missions import register_boot_autonomy
    from nomorals.storage.db import WAL_CHECKPOINT_JOB

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock, overdue=False)
    calls, notify = _notify_sink()
    live = _FakeScheduler([{"id": "job-0", "name": WAL_CHECKPOINT_JOB,
                            "spec": "every 1h"}])
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           scheduler=live, clock=clock)
    assert live.list_jobs() == []


# ── the mission producer ──────────────────────────────────────────────


def test_producer_schedules_mission_for_due_commitment(tmp_path):
    """A due commitment becomes a real mission on the unified queue:
    one plan step (notify the owner), the commitment id in the
    metadata, the goal naming the promise."""
    from nomorals.autonomy.missions import produce_commitment_missions

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    ledger, due, _ = _ledger_with_due(clock=clock)
    produced = produce_commitment_missions(stack, ledger, now=clock())
    assert len(produced) == 1
    mission = stack.scheduler.get_mission(produced[0])
    assert mission["status"] == "pending"
    assert mission["metadata"]["commitment_id"] == due.id
    assert mission["organ"] == "commitments"
    assert len(mission["plan"]) == 1
    assert mission["plan"][0]["handler"] == "notify_owner"
    assert "Ada" in mission["goal"] and "call back" in mission["goal"]


def test_producer_dedupes_pending_mission(tmp_path):
    """Producing twice does not schedule twice: a pending mission for
    the same commitment blocks a second one."""
    from nomorals.autonomy.missions import produce_commitment_missions

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    ledger, _, _ = _ledger_with_due(clock=clock)
    assert len(produce_commitment_missions(stack, ledger, now=clock())) == 1
    assert produce_commitment_missions(stack, ledger, now=clock()) == []
    assert len(stack.scheduler.pending_missions()) == 1


def test_producer_dedupes_across_stacks_via_refresh(tmp_path):
    """Two runtimes, one queue file: the second stack refreshes the
    durable queue before producing, sees the first stack's pending
    mission, and schedules nothing — no duplicate nudge mission."""
    from nomorals.autonomy.missions import produce_commitment_missions

    clock = _Clock()
    stack_a = _stack(tmp_path, clock)
    stack_b = _stack(tmp_path, clock)  # built BEFORE A produces: B's
    # in-memory queue is empty, so only a refresh saves it from
    # scheduling a duplicate.
    ledger, _, _ = _ledger_with_due(clock=clock)
    assert len(produce_commitment_missions(stack_a, ledger, now=clock())) == 1

    assert produce_commitment_missions(stack_b, ledger, now=clock()) == []
    assert len(_stack(tmp_path, clock).scheduler.all_missions()) == 1


def test_producer_skips_done_recently_then_reproduces(tmp_path):
    """A completed nudge is not repeated for a day — then a fresh one
    is legitimate: the commitment is still open and still due, so the
    owner gets reminded again, once a day at most."""
    from nomorals.autonomy.missions import (
        produce_commitment_missions,
        register_boot_autonomy,
    )

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    produced = produce_commitment_missions(stack, ledger, now=clock())
    assert len(produced) == 1
    stack.scheduler.run_due(now=clock())  # producer tick: no-op (deduped)
    stack.scheduler.run_due(now=clock())  # the mission fires: done
    assert stack.scheduler.get_mission(produced[0])["status"] == "done"
    assert produce_commitment_missions(stack, ledger, now=clock()) == []
    clock.advance(25 * 3600)
    assert len(produce_commitment_missions(stack, ledger, now=clock())) == 1


def test_producer_skips_nothing_due(tmp_path):
    """Nothing due, nothing produced — and a kept commitment never
    produces, however near its deadline was."""
    from nomorals.autonomy.missions import produce_commitment_missions

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    ledger, due, far = _ledger_with_due(clock=clock, overdue=False,
                                        future=True)
    assert produce_commitment_missions(stack, ledger, now=clock()) == []

    ledger2, due2, _ = _ledger_with_due(clock=clock)
    ledger2.mark_kept(due2.id)
    assert produce_commitment_missions(stack, ledger2, now=clock()) == []


# ── the handler, registered at boot ───────────────────────────────────


def test_notify_handler_delivers_and_completes(tmp_path):
    """The boot-registered ``notify_owner`` handler runs the mission's
    step for real: the owner gets the nudge text, the mission
    completes, and a second tick does not notify again."""
    from nomorals.autonomy.missions import register_boot_autonomy

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    # run_due runs missions BEFORE ticks: the first pass produces the
    # mission (producer tick) without running it; the second runs it.
    first = stack.scheduler.run_due(now=clock())
    assert not any(r.get("kind") == "mission" for r in first)
    assert len(calls) == 0
    assert len(stack.scheduler.pending_missions()) == 1
    second = stack.scheduler.run_due(now=clock())
    assert any(r.get("kind") == "mission" and r.get("status") == "done"
               for r in second)
    assert len(calls) == 1
    kind, title, body = calls[0]
    assert kind == "commitment"
    assert "Ada" in body and "call back" in body
    assert "Overdue" in body  # deadline is in the past
    third = stack.scheduler.run_due(now=clock())
    assert not any(r.get("kind") == "mission" for r in third)
    assert len(calls) == 1  # exactly once


def test_producer_tick_registered_at_boot(tmp_path):
    """Boot registers the producer itself as an organ tick: the scan
    runs hourly on the unified driver, not only when someone calls it
    by hand."""
    from nomorals.autonomy.missions import register_boot_autonomy

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock, overdue=False)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)
    tick = stack.scheduler._ticks["commitment-producer"]
    assert tick["interval"] == 3600.0
    assert tick["organ"] == "commitments"


# ── end to end through the live driver ────────────────────────────────


def test_mission_fires_once_through_tick_tool(tmp_path):
    """The full live path: the boot stack sits on the runtime, the
    production ``__unified_scheduler_tick__`` tool fires the due
    mission exactly once, and the next tick fires nothing for it."""
    from nomorals.autonomy.missions import register_boot_autonomy

    clock = _Clock()
    stack = _stack(tmp_path, clock)
    db = _FakeDb()
    ledger, _, _ = _ledger_with_due(clock=clock)
    calls, notify = _notify_sink()
    register_boot_autonomy(stack, db=db, ledger=ledger, notify_fn=notify,
                           clock=clock)

    ctx = _scheduler_ctx(tmp_path)
    ctx.extras["autonomy_stack"] = stack
    tools = _capture_tools(ctx)
    # First pass: the producer tick schedules the mission (missions
    # run before ticks in run_due, so it is produced, not yet run).
    first = tools["__unified_scheduler_tick__"]()
    assert first["ok"] is True
    assert len(calls) == 0
    assert len(stack.scheduler.pending_missions()) == 1
    # Second pass: the mission fires exactly once.
    second = tools["__unified_scheduler_tick__"]()
    assert second["ok"] is True
    assert len(calls) == 1
    # Third pass: the mission is done, nothing re-fires.
    tools["__unified_scheduler_tick__"]()
    assert len(calls) == 1  # the mission is done: never re-fired


def test_tick_tool_fresh_stack_skips_missing_handler_honestly(tmp_path):
    """Restart without re-registration: a fresh stack sees the durable
    mission but has no handler — the step is recorded as skipped, the
    mission completes, nothing raises into the scheduler."""
    from nomorals.autonomy.missions import produce_commitment_missions

    clock = _Clock()
    producer = _stack(tmp_path, clock)
    ledger, _, _ = _ledger_with_due(clock=clock)
    produced = produce_commitment_missions(producer, ledger, now=clock())
    assert len(produced) == 1

    ctx = _scheduler_ctx(tmp_path, home=tmp_path)  # boot never ran
    tools = _capture_tools(ctx)
    result = tools["__unified_scheduler_tick__"]()
    assert result["ok"] is True
    # The producer stack's in-memory copy is stale — re-read the
    # durable queue to see what the fresh stack did.
    from nomorals.autonomy.wiring import build_stack

    mission = build_stack(tmp_path).scheduler.get_mission(produced[0])
    assert mission["status"] == "done"
    assert any(r.get("status") == "done" for r in result["results"])
