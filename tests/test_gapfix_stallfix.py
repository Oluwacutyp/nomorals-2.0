"""Stall fix (2026-11-19, owner screenshot 23:59 MST): scheduled jobs
wedging live — news poll timing out after 300s in Database.query_rows
twice (1 → 2 abandoned workers), motivation morning after 900s in
Path.open, then a restart.  REGRESSION-FIRST: the v89 DB-stall fix
(796aa152) bound the failure-clusters scan and pruned news_items, but
the prune itself was the wedge, and the scheduler never saw its own
abandoned workers — so the next poll stacked a second ghost behind
the first.

These pins prove the unbounded paths on the base code, then the
bounds on the fix.  Every one of them fails on c639a4d7 and passes
after the fix — the failing-first run is recorded in SURVEY.md.
"""

import json
import threading
import time
from types import SimpleNamespace

from nomorals.agents.news import NewsAgent
from nomorals.agents.news_jobs import ensure_news_jobs
from nomorals.agents.scheduler import Scheduler
from nomorals.agents.teachings import (
    CANDIDATE_TTL_S,
    TeachingsLedger,
)
from nomorals.storage.db import Database

# ── harness ──────────────────────────────────────────────────────────

def _ctx():
    """A scheduler-shaped context on an in-memory migrated DB, exactly
    the shape tests/test_scheduler_pass11.py builds."""
    ctx = SimpleNamespace(
        db=Database(":memory:"), dbpath=None,
        settings=SimpleNamespace(), profile="normal",
        home=None, ledger=None, notifications=None,
        event=SimpleNamespace(emit=lambda *a, **k: None),
        tools=SimpleNamespace(
            call=lambda *a, **k: SimpleNamespace(ok=True, value="ok",
                                                 error=None)),
    )
    ctx.db.migrate()  # Database.migrate(): applies every migration
    return ctx


class _SlowTool:
    """A tool payload that sleeps — the wedged job.  The count is the
    proof: a second firing while the first worker is still alive is a
    second live call running (accumulation)."""

    def __init__(self, sleep_s: float = 5.0) -> None:
        self.sleep_s = sleep_s
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        time.sleep(self.sleep_s)
        return SimpleNamespace(ok=True, value="ok", error=None)


def _slow_job(sched: Scheduler, tool: _SlowTool, *,
              overlap_policy: str = "skip",
              run_timeout_s: float = 0.5) -> str:
    """Register a skip-policy tool job due now; return its id."""
    ctx = sched.context
    ctx.tools.call = tool
    job = sched.add(
        "slow job", "every 30m", "tool",
        {"tool": "slow_probe", "args": {}},
        routine=True, overlap_policy=overlap_policy,
        run_timeout_s=run_timeout_s)
    sched.db.execute(
        "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
        (time.time() - 1.0, job["id"]))
    return str(job["id"])


def _join_ghosts(sched: Scheduler, job_id: str, timeout_s: float = 12.0) -> None:
    """Join every abandoned worker for a job so no test leaks a live
    thread into the next one."""
    deadline = time.monotonic() + timeout_s
    for entry in sched.abandoned_workers(job_id):
        worker = entry.get("thread")
        if worker is not None:
            worker.join(max(0.0, deadline - time.monotonic()))


# ── (b) accumulation: a live abandoned worker blocks the next firing ──

def test_abandoned_worker_blocks_next_firing():
    """The 2026-11-19 screenshot, reproduced: news poll timed out at
    9:55 PM (1 abandoned worker), then AGAIN at 9:59 PM (2 abandoned
    workers).  The second poll ran while the first worker was still
    alive — _active had already dropped the job, and the overlap check
    never looked at _abandoned.  With the fix the second firing is
    skipped (recorded as skipped), the tool is NOT called again, and
    the abandoned count stays at 1."""
    sched = Scheduler(_ctx())
    tool = _SlowTool(sleep_s=5.0)
    job_id = _slow_job(sched, tool)
    try:
        sched.tick()
        assert tool.calls == 1
        assert sched.abandoned_count(job_id) == 1
        # The next firing is due now, while the ghost still lives.
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 1.0, job_id))
        sched.tick()
        assert tool.calls == 1, (
            "second firing ran while the first worker was still alive")
        assert sched.abandoned_count(job_id) == 1, (
            "abandoned workers must not pile up across firings")
        skipped = sched.db.query(
            "SELECT * FROM schedule_runs WHERE job_id = ? "
            "AND status = 'skipped' ORDER BY started_at DESC LIMIT 1",
            (job_id,))
        assert skipped, "the blocked firing must be recorded as skipped"
        assert "abandoned" in (skipped[0].get("result") or ""), (
            "the skip reason must name the live abandoned worker")
    finally:
        _join_ghosts(sched, job_id)


def test_tick_reinterrupts_live_abandoned_worker():
    """One prod at abandonment is not enough: the worker's own error
    handling catches the aborted step and moves on to the next DB call,
    where it can wedge again.  A live ghost is prodded on every tick
    until it unwinds.  Here the prod target is the ORIGINAL worker —
    counted by ident, so a second worker on the base code cannot fake
    the count."""
    sched = Scheduler(_ctx())
    tool = _SlowTool(sleep_s=5.0)
    job_id = _slow_job(sched, tool)
    prods: list[int] = []
    orig_interrupt = sched.db.interrupt_thread

    def counting_interrupt(ident: int) -> bool:
        prods.append(ident)
        return orig_interrupt(ident)

    sched.db.interrupt_thread = counting_interrupt
    try:
        sched.tick()
        ghosts = sched.abandoned_workers(job_id)
        assert len(ghosts) == 1
        ghost_ident = ghosts[0]["thread"].ident
        assert prods.count(ghost_ident) == 1, (
            "abandonment itself must prod the worker once")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 1.0, job_id))
        sched.tick()
        assert tool.calls == 1
        assert prods.count(ghost_ident) >= 2, (
            "a live ghost must be re-interrupted on the next tick")
    finally:
        sched.db.interrupt_thread = orig_interrupt
        _join_ghosts(sched, job_id)


def test_news_poll_registers_overlap_skip():
    """ensure_news_jobs must register 'news poll' with
    overlap_policy=skip: the poll's own gate only ever saw _active
    runs, so an abandoned worker looked like a free slot.  On the
    base code the job registers 'concurrent' (the default) — the
    exact hole the screenshot shows."""
    sched = Scheduler(_ctx())
    ensure_news_jobs(sched)
    rows = [j for j in sched.list_jobs() if j.get("name") == "news poll"]
    assert len(rows) == 1
    assert rows[0].get("overlap_policy") == "skip"


def test_news_poll_reconciles_persisted_overlap_policy():
    """The row on the owner's box was born months ago with the default
    'concurrent' — add() only ever sets the policy on a FRESH row, so
    merely changing the registrar's add() call would never reach it.
    ensure_news_jobs must reconcile the persisted row to 'skip' on
    every ensure, so the fix lands on the next boot with no restart
    and no manual job edit."""
    sched = Scheduler(_ctx())
    # The old registration, as it stands on the live box.
    sched.add("news poll", "every 30m", "tool",
              {"tool": "news_poll", "args": {}}, routine=True)
    before = [j for j in sched.list_jobs()
              if j.get("name") == "news poll"]
    assert before[0].get("overlap_policy") == "concurrent"
    ensure_news_jobs(sched)
    after = [j for j in sched.list_jobs()
             if j.get("name") == "news poll"]
    assert len(after) == 1, "ensure must not duplicate the job"
    assert after[0].get("overlap_policy") == "skip"
    # Idempotent: a second ensure leaves it skipped, still one job.
    ensure_news_jobs(sched)
    again = [j for j in sched.list_jobs()
             if j.get("name") == "news poll"]
    assert len(again) == 1
    assert again[0].get("overlap_policy") == "skip"


# ── (a) the news-prune wedge: one unbounded DELETE ───────────────────

def _seed_news(db: Database, *, settled: int, undelivered: int,
               old_days: float = 90.0) -> None:
    """Seed news_items exactly the way probe_scale.py does (same
    columns, same states)."""
    old = time.time() - old_days * 86400.0
    rows = []
    for i in range(settled):
        rows.append((f"s{i}", f"feed-{i % 86}", f"Title {i}",
                     f"https://example.com/s{i}", "summary",
                     str(old - i), old - i, "pushed"))
    for i in range(undelivered):
        rows.append((f"u{i}", f"feed-{i % 86}", f"Title {i}",
                     f"https://example.com/u{i}", "summary",
                     str(old - i), old - i, "undelivered"))
    db.executemany(
        "INSERT INTO news_items (id, source, title, url, summary, "
        "published, created_at, delivery_state) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)", rows)


def test_prune_news_items_deletes_in_bounded_batches():
    """The 500k-row probe (stallfix-verify/scale-500k.log): the old
    single DELETE took 25.77s at 500k rows and held the writer actor
    for its whole duration.  The fix deletes in 2000-row batches
    (max 20000 per call).  Seeded here at 5000 settled + 100 live
    undelivered rows: no single DELETE may exceed 2000 rows, the
    total deleted is all 5000, and the undelivered rows survive."""
    ctx = _ctx()
    _seed_news(ctx.db, settled=5000, undelivered=100)
    agent = NewsAgent(ctx)
    deletes: list[int] = []
    orig_execute = ctx.db.execute

    def counting_execute(sql, params=()):
        cur = orig_execute(sql, params)
        if str(sql).lstrip().upper().startswith("DELETE"):
            deletes.append(int(cur.rowcount or 0))
        return cur

    ctx.db.execute = counting_execute
    try:
        dropped = agent.prune_news_items(older_than_days=60.0)
    finally:
        ctx.db.execute = orig_execute
    assert dropped == 5000
    assert deletes, "prune must issue DELETEs"
    assert max(deletes) <= 2000, (
        f"a single DELETE deleted {max(deletes)} rows — the wedge is "
        "back (batched prune caps at 2000)")
    assert sum(deletes) == 5000
    left = ctx.db.query(
        "SELECT COUNT(*) AS n FROM news_items WHERE "
        "delivery_state = 'undelivered'")
    assert left[0]["n"] == 100


def test_prune_news_items_stops_at_call_budget():
    """A huge backlog drains over successive hourly polls instead of
    one giant statement: the per-call cap (20000 rows) bounds any
    single prune however large the table is."""
    ctx = _ctx()
    _seed_news(ctx.db, settled=25000, undelivered=0)
    agent = NewsAgent(ctx)
    dropped = agent.prune_news_items(older_than_days=60.0)
    assert dropped == 20000
    left = ctx.db.query("SELECT COUNT(*) AS n FROM news_items")
    assert left[0]["n"] == 5000
    # The remainder drains on the next call.
    assert agent.prune_news_items(older_than_days=60.0) == 5000


# ── (a) the motivation-morning wedge: unbounded teachings reads ──────

def _seed_teachings(db: Database, *, live: int, expired: int,
                    served: int) -> None:
    now = time.time()
    db.executemany(
        "INSERT OR REPLACE INTO teachings_candidates "
        "(item_key, item_json, fetched_at) VALUES (?, ?, ?)",
        [(f"live{i}", json.dumps({"_key": f"live{i}", "title": "t"}),
          now - 60.0) for i in range(live)]
        + [(f"old{i}", json.dumps({"_key": f"old{i}", "title": "t"}),
            now - 48 * 3600.0) for i in range(expired)],
    )
    db.executemany(
        "INSERT INTO teachings_items (id, kind, title, body, theme, "
        "source_type, source_ref, source_label, passage, item_key, "
        "created_at, served_json) "
        "VALUES (?, 'teaching', 't', 'b', '', '', '', '', '', ?, ?, '[]')",
        [(f"item{i}", f"item{i}", now - i * 3600.0)
         for i in range(served)],
    )


def test_teachings_candidate_rows_live_only():
    """The old _candidate_rows fetched the whole candidates table with
    no WHERE and no LIMIT, and the callers then dropped every row
    older than the 24h TTL in Python — a full table fetch on every
    stock/pick, growing forever.  With the fix the TTL filter is in
    SQL (seek on the fetched_at index): only the live rows come
    back."""
    ctx = _ctx()
    ledger = TeachingsLedger(ctx)
    _seed_teachings(ctx.db, live=100, expired=2900, served=0)
    rows = ledger._candidate_rows()
    assert len(rows) == 100, (
        f"_candidate_rows returned {len(rows)} rows — expired "
        "candidates must be filtered in SQL, not fetched and dropped")
    plan = ctx.db.explain(
        "SELECT item_json, fetched_at FROM teachings_candidates "
        "WHERE fetched_at >= ?",
        (time.time() - CANDIDATE_TTL_S,))
    detail = " ".join(r.get("detail", "") for r in plan)
    assert "USING INDEX" in detail, (
        f"candidate read must be an index seek, got: {detail}")


def test_teachings_served_items_bounded():
    """The old served_items fetched the whole served table newest-first
    with no LIMIT — every cooldown check paid a full scan of all
    serves ever (teachings_items has NO indexes and no bound).  With
    the fix the read is capped at the newest 2000, and the cap is
    newest-first so the 30-day cooldown window sits far inside it."""
    ctx = _ctx()
    ledger = TeachingsLedger(ctx)
    _seed_teachings(ctx.db, live=0, expired=0, served=2500)
    items = ledger.served_items()
    assert len(items) <= 2000, (
        f"served_items returned {len(items)} rows — unbounded")
    assert len(items) == 2000
    # Newest first: the first row is the most recently served.
    assert float(items[0]["created_at"]) > float(items[-1]["created_at"])
    plan = ctx.db.explain(
        "SELECT * FROM teachings_items ORDER BY created_at DESC LIMIT ?",
        (2000,))
    detail = " ".join(r.get("detail", "") for r in plan)
    assert "USING INDEX idx_teachings_items_created" in detail, (
        f"served read must use the created_at index, got: {detail}")
