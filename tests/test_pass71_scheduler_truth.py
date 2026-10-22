"""PASS #71: scheduler truth — routine is a flag, errors record as failures.

Two defects banked from directed Pass 3 (the immune-sampler spam):

1. The scheduler classified "routine" jobs (whose successful runs stay
   silent) by SUBSTRING MATCHING ON THE JOB NAME ('tick' / 'heartbeat' /
   'sweep'). The immune sampler spammed the owner because its original
   name missed the convention. Classification is now a first-class
   ``routine`` field on the job definition; persisted pre-flag rows are
   backfilled ONCE with exactly what the name rule computed for them.

2. A tool job whose tool VALUE declared failure (``{"ok": False,
   "error": ...}`` — the codebase-wide tool convention) was recorded
   as a SUCCESSFUL run, because only a raised exception (or a registry
   Err) made the run fail. Declared failures now record exactly like
   raised ones: ``schedule_runs.ok = 0``, ``last_result`` starts with
   "job failed", the error text preserved.
"""

from types import SimpleNamespace

import pytest

from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database


# ── harness ──────────────────────────────────────────────────────────

class _Tools:
    """Duck-typed tool registry: per-tool canned Outcomes."""

    def __init__(self, outcomes):
        self._outcomes = outcomes

    def call(self, tool, capabilities=None, **args):
        outcome = self._outcomes[tool]
        if callable(outcome):
            return outcome(**args)
        return outcome


def _ok(value):
    return SimpleNamespace(ok=True, value=value, error=None)


def _err(message):
    return SimpleNamespace(ok=False, value=None,
                           error=SimpleNamespace(message=message))


def _ctx(tools=None):
    db = Database(":memory:")
    db.migrate()
    if tools is None:
        tools = _Tools({"echo": _ok({"ok": True, "said": "hi"})})
    return SimpleNamespace(db=db, tools=tools,
                           settings=SimpleNamespace(), extras={})


class _Notifier:
    """Recording notifier stub (delivered nowhere, remembers all)."""

    def __init__(self, delivered=True):
        self.calls = []
        self._delivered = delivered

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        if self._delivered:
            return {"delivered": True, "channel": "test"}
        return {"delivered": False, "delivery_state": "pending"}


def _sched(ctx):
    sched = Scheduler(ctx)
    sched.notifier = _Notifier()
    return sched


def _job(sched, name):
    rows = [j for j in sched.list_jobs() if j.get("name") == name]
    assert len(rows) == 1, f"expected one job named {name!r}, got {rows}"
    return rows[0]


def _schedule_alerts(sched):
    return [c for c in sched.notifier.calls if c["kind"] == "schedule"]


# ── (a/b) classification reads the flag, never the name ─────────────

def test_routine_flag_silences_success_despite_plain_name():
    sched = _sched(_ctx())
    job = sched.add("quarterly report", "every 1h", "tool",
                    {"tool": "echo"}, routine=True)
    out = sched.run_now(job["id"])
    assert out["ok"] is True
    assert _schedule_alerts(sched) == [], \
        "routine-flagged job paged the owner on success"


def test_non_routine_job_with_routine_name_still_alerts():
    sched = _sched(_ctx())
    # the name carries every legacy routine substring — irrelevant now
    job = sched.add("nightly sweep tick heartbeat", "every 1h", "tool",
                    {"tool": "echo"})
    out = sched.run_now(job["id"])
    assert out["ok"] is True
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1 and alerts[0]["title"].startswith("✅"), \
        "non-routine job was silenced by its name alone"


def test_default_for_new_job_is_non_routine():
    sched = _sched(_ctx())
    job = sched.add("plain job", "every 1h", "tool", {"tool": "echo"})
    assert _job(sched, "plain job")["routine"] is False
    out = sched.run_now(job["id"])
    assert out["ok"] is True
    assert len(_schedule_alerts(sched)) == 1


# ── (c) renaming changes nothing ─────────────────────────────────────

def test_rename_does_not_change_classification():
    ctx = _ctx()
    sched = _sched(ctx)
    job = sched.add("ops note", "every 1h", "tool", {"tool": "echo"},
                    routine=True)
    sched.run_now(job["id"])
    assert _schedule_alerts(sched) == []
    # a rename at the storage layer (there is no rename API — the name
    # is just a column) must not reclassify the job either way
    with ctx.db.transaction():
        ctx.db.execute("UPDATE schedule_jobs SET name = ? WHERE id = ?",
                       ("ops tick sweep heartbeat renamed", job["id"]))
    sched.run_now(job["id"])
    assert _schedule_alerts(sched) == [], \
        "renaming a routine job made its successes page again"


# ── (d) day-one behavior: built-ins + persisted rows unchanged ───────

#: the pre-#71 name rule, reproduced here as history: a job was
#: routine iff its lowercase name contained tick / heartbeat / sweep.
def _legacy_routine(name):
    lname = name.lower()
    return "tick" in lname or "heartbeat" in lname or "sweep" in lname


BUILTIN_NAMES = [
    # registered by ensure-functions / runtime wiring across the repo
    "agency-shift", "agent modes tick", "immune sampler tick",
    "watchers sweep", "rooms tick", "presence-heartbeat",
    "improvement tick", "research organ tick", "wisdom organ tick",
    "monitor tick", "improve-evolution-tick", "improve-synthesis-scan",
    "morning-pulse", "afternoon-pulse", "evening-pulse", "briefing",
    "research loop", "persona rebuild", "persona curate",
    "memory-consolidation", "memory consolidation", "proxy_refresh",
    "mining session", "finance alerts", "finance digest",
    "cognitive loop",
    # the legacy name Pass 3 replaced (still in old databases)
    "immune baseline sampler",
]


def test_persisted_pre_flag_rows_backfilled_to_legacy_classification():
    db = Database(":memory:")
    db.migrate()
    # rows as a pre-#71 database holds them: no routine column exists
    # until a Scheduler is constructed on this database.
    cols = [r["name"] for r in db.query("PRAGMA table_info(schedule_jobs)")]
    assert "routine" not in cols
    import json as _json
    import time as _time
    for i, name in enumerate(BUILTIN_NAMES):
        db.execute(
            "INSERT INTO schedule_jobs (id, name, kind, spec, "
            "payload_kind, payload, enabled, next_run, created_at, "
            "updated_at) VALUES (?, ?, 'every', '3600', 'tool', ?, 1, "
            "?, ?, ?)",
            (f"legacy-{i}", name, _json.dumps({"tool": "echo"}),
             _time.time() + 3600, _time.time(), _time.time()))
    ctx = SimpleNamespace(db=db, tools=_Tools({}), settings=SimpleNamespace(),
                          extras={})
    sched = Scheduler(ctx)  # construction ensures + backfills, once
    for name in BUILTIN_NAMES:
        row = _job(sched, name)
        assert row["routine"] is _legacy_routine(name), \
            f"{name!r}: flag {row['routine']} != legacy {_legacy_routine(name)}"
    # the backfill is one-time: a later rename never re-derives
    with db.transaction():
        db.execute("UPDATE schedule_jobs SET name = ? WHERE name = ?",
                   ("renamed baseline sampler", "immune baseline sampler"))
    row = _job(sched, "renamed baseline sampler")
    assert row["routine"] is False


def test_ensure_functions_register_the_legacy_classification():
    from nomorals.agency.wiring import ensure_agency_shift_job
    from nomorals.agents.modes import ensure_agent_modes_tick_job
    from nomorals.agents.scheduler import ensure_persona_jobs
    from nomorals.immune.wiring import ensure_immune_sampler_job
    from nomorals.memory.cadence import ensure_consolidation_job
    from nomorals.workspace.rooms import ensure_rooms_tick_job
    from nomorals.agents.watchers import ensure_sweeper_job
    from nomorals.autonomy.presence import ensure_heartbeat_job

    ctx = _ctx()
    sched = Scheduler(ctx)
    ensure_agency_shift_job(sched)
    ensure_agent_modes_tick_job(sched)
    ensure_immune_sampler_job(sched)
    ensure_rooms_tick_job(ctx)
    ensure_sweeper_job(ctx)
    ensure_consolidation_job(ctx, scheduler_factory=Scheduler)
    ensure_persona_jobs(ctx)
    ensure_heartbeat_job(sched)
    for name in ("agency-shift", "agent modes tick", "immune sampler tick",
                 "rooms tick", "watchers sweep", "memory-consolidation",
                 "persona rebuild", "persona curate", "presence-heartbeat"):
        row = _job(sched, name)
        assert row["routine"] is _legacy_routine(name), \
            f"{name!r} registered routine={row['routine']}, " \
            f"legacy classification is {_legacy_routine(name)}"


# ── (e) declared failures record as failures ─────────────────────────

def _run_one(tools, payload_kind="tool", payload=None, name="job one"):
    ctx = _ctx(tools)
    sched = _sched(ctx)
    job = sched.add(name, "every 1h", payload_kind,
                    payload or {"tool": "echo"})
    out = sched.run_now(job["id"])
    runs = sched.recent_runs(job["id"])
    return sched, job, out, runs


def test_tool_value_ok_false_records_failed_with_error_text():
    tools = _Tools({"echo": _ok({"ok": False,
                                 "error": "kaboom: signature mismatch"})})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is False
    assert len(runs) == 1 and runs[0]["ok"] is False
    row = _job(sched, "job one")
    assert row["last_result"].startswith("job failed")
    assert "kaboom: signature mismatch" in row["last_result"]
    assert "kaboom: signature mismatch" in runs[0]["result"]


def test_tool_value_ok_false_with_reason_only_records_failed():
    tools = _Tools({"echo": _ok({"ok": False, "reason": "no capacity"})})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is False
    assert runs[0]["ok"] is False
    assert "no capacity" in _job(sched, "job one")["last_result"]


def test_tool_value_bare_error_dict_records_failed():
    # cipher/deals-style argument errors: {"error": ...} with no ok key
    tools = _Tools({"echo": _ok({"error": "Unknown action: frobnicate"})})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is False
    assert runs[0]["ok"] is False
    assert "Unknown action: frobnicate" in _job(sched, "job one")["last_result"]


def test_tool_value_failure_alerts_like_a_raised_failure():
    tools = _Tools({"echo": _ok({"ok": False, "error": "boom"})})
    sched, job, out, runs = _run_one(tools)
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1 and alerts[0]["title"].startswith("❌")


def test_routine_declared_failure_records_failed_without_generic_alert():
    # A routine job's value-declared failure is the job's own report:
    # routine systems publish their own events (the immune sampler's
    # tick errors are the canonical case — its tool already notified,
    # with specifics). The run RECORDS failed; the generic scheduler
    # alert must not double-page it.
    tools = _Tools({"echo": _ok({"ok": False, "error": "tick boom"})})
    ctx = _ctx(tools)
    sched = _sched(ctx)
    job = sched.add("quiet bookkeeping", "every 1h", "tool",
                    {"tool": "echo"}, routine=True)
    out = sched.run_now(job["id"])
    assert out["ok"] is False
    runs = sched.recent_runs(job["id"])
    assert runs[0]["ok"] is False
    assert "tick boom" in _job(sched, "quiet bookkeeping")["last_result"]
    assert _schedule_alerts(sched) == []


def test_routine_raised_failure_still_alerts_on_transition():
    # machinery failures have no other reporter — unchanged behavior
    tools = _Tools({"echo": _err("registry blew up")})
    ctx = _ctx(tools)
    sched = _sched(ctx)
    job = sched.add("quiet bookkeeping", "every 1h", "tool",
                    {"tool": "echo"}, routine=True)
    out = sched.run_now(job["id"])
    assert out["ok"] is False
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1 and alerts[0]["title"].startswith("❌")


def test_raised_tool_failure_still_records_failed():
    tools = _Tools({"echo": _err("registry blew up")})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is False
    assert runs[0]["ok"] is False
    assert "registry blew up" in _job(sched, "job one")["last_result"]


def test_tool_success_value_records_successful():
    tools = _Tools({"echo": _ok({"ok": True, "sampled": True,
                                 "learned": True})})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is True
    assert runs[0]["ok"] is True
    assert not _job(sched, "job one")["last_result"].startswith("job failed")


def test_command_nonzero_exit_records_failed_with_output():
    tools = _Tools({"shell_run": _ok({"exit_code": 3,
                                      "stdout": "bad stuff happened",
                                      "stderr": ""})})
    sched, job, out, runs = _run_one(
        tools, payload_kind="command", payload={"command": "make it so"},
        name="cmd job")
    assert out["ok"] is False
    assert runs[0]["ok"] is False
    last = _job(sched, "cmd job")["last_result"]
    assert last.startswith("job failed")
    assert "bad stuff happened" in last


def test_command_zero_exit_records_successful():
    tools = _Tools({"shell_run": _ok({"exit_code": 0, "stdout": "fine",
                                      "stderr": ""})})
    sched, job, out, runs = _run_one(
        tools, payload_kind="command", payload={"command": "true"},
        name="cmd ok")
    assert out["ok"] is True
    assert runs[0]["ok"] is True


# ── (f) ambiguous shapes stay recorded as before ─────────────────────

def test_ok_true_with_error_note_stays_successful():
    # the tool declared success overall; the error key is a sub-note
    tools = _Tools({"echo": _ok({"ok": True,
                                 "error": "one sub-fetch failed"})})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is True
    assert runs[0]["ok"] is True


def test_plain_dict_without_ok_or_error_stays_successful():
    tools = _Tools({"echo": _ok({"count": 3, "items": [1, 2, 3]})})
    sched, job, out, runs = _run_one(tools)
    assert out["ok"] is True
    assert runs[0]["ok"] is True


def test_command_value_without_exit_code_stays_successful():
    tools = _Tools({"shell_run": _ok({"stdout": "no code reported"})})
    sched, job, out, runs = _run_one(
        tools, payload_kind="command", payload={"command": "mystery"},
        name="cmd ambiguous")
    assert out["ok"] is True
    assert runs[0]["ok"] is True


def test_message_stored_undelivered_stays_successful():
    # delivery state is the notifier/redelivery queue's business: a
    # stored (pending) message is not a failed run — unchanged behavior
    ctx = _ctx()
    sched = Scheduler(ctx)
    sched.notifier = _Notifier(delivered=False)
    job = sched.add("note to self", "every 1h", "message",
                    {"text": "remember this"})
    out = sched.run_now(job["id"])
    assert out["ok"] is True
    runs = sched.recent_runs(job["id"])
    assert runs[0]["ok"] is True
