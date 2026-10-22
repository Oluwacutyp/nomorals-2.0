"""PASS #84: the routine flag becomes reachable on the schedule surfaces.

Pass #71 made ``routine`` a stored flag on schedule jobs (routine jobs'
successes stay quiet; failures still surface) — but named its own
residue: "/schedule add surfaces don't expose the flag yet."  The opt-in
existed in storage and scheduler behavior while NO creation surface
could set it.  This pass exposes it end to end:

* ``Scheduler.set_routine`` — the edit seam (mirrors ``set_enabled``).
* chat ``/schedule add`` — trailing ``routine`` flag / ``routine on|off``
  option (the ``heavy`` idiom), invalid values rejected honestly.
* chat ``/schedule list`` — routine jobs carry a ``routine`` marker.
* chat ``/schedule routine <name> <on|off>`` — toggle an existing job.
* CLI ``nm schedule`` — the same add option, list marker, and
  ``routine`` verb on the mirrored surface.
* NL reaches the flag through the same command layer (no registry
  schedule tool exists — verified in the #84 survey); no keyword
  shortcuts were added anywhere.
"""

import io
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

import pytest

from nomorals.agents.scheduler import Scheduler, parse_routine_word
from nomorals.storage.db import Database


# ── harness (mirrors test_pass71_scheduler_truth) ────────────────────

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

    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


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


from nomorals.agents.partner.runtime_schedule import RuntimeScheduleMixin


class _ChatStub(RuntimeScheduleMixin):
    """The smallest PartnerRuntime stand-in the schedule mixin needs:
    a preset ``_scheduler`` (so ``_scheduler_or_build`` returns it)
    plus the context/gateway attributes the mixin reads."""

    def __init__(self, scheduler):
        self._scheduler = scheduler
        self.context = scheduler.context
        self.gateway = None

    def control(self, tail):
        return self._control_schedule(tail)


def _chat(sched):
    return _ChatStub(sched)


# ── word parser ──────────────────────────────────────────────────────

def test_parse_routine_word():
    for word in ("on", "ON", "true", "yes", "1"):
        assert parse_routine_word(word) is True
    for word in ("off", "Off", "false", "no", "0"):
        assert parse_routine_word(word) is False
    for word in ("banana", "", "onnn", None):
        assert parse_routine_word(word) is None


# ── (a) create via each surface ──────────────────────────────────────

def test_chat_add_bare_routine_flag_stores_true():
    sched = _sched(_ctx())
    reply = _chat(sched).control(
        "add quietbooks every 1h tool echo routine")
    assert "scheduled quietbooks" in reply
    assert "routine" in reply  # the confirmation names the quiet marker
    assert _job(sched, "quietbooks")["routine"] is True


def test_chat_add_routine_on_off_values():
    sched = _sched(_ctx())
    _chat(sched).control("add onjob every 1h tool echo routine on")
    assert _job(sched, "onjob")["routine"] is True
    _chat(sched).control("add offjob every 1h tool echo routine off")
    assert _job(sched, "offjob")["routine"] is False


def test_chat_add_without_routine_defaults_false():
    sched = _sched(_ctx())
    _chat(sched).control("add plainjob every 1h tool echo")
    assert _job(sched, "plainjob")["routine"] is False


def test_chat_add_routine_combines_with_other_options():
    sched = _sched(_ctx())
    reply = _chat(sched).control(
        "add combo every 1h tool echo heavy routine")
    assert "scheduled combo" in reply
    row = _job(sched, "combo")
    assert row["routine"] is True and row["heavy"] is True


def test_api_add_default_stays_false():
    sched = _sched(_ctx())
    sched.add("api job", "every 1h", "tool", {"tool": "echo"})
    assert _job(sched, "api job")["routine"] is False


# ── CLI surface ──────────────────────────────────────────────────────

def _cli_ctx():
    from nomorals.agents.context import build_context
    return build_context(db=Database(":memory:"), with_tools=True,
                         with_router=False, with_memory=False)


def _cli(argv, ctx):
    from nomorals.cmdline.commands.schedule import _cmd_schedule
    from nomorals.cmdline.parser import _parser
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _cmd_schedule(_parser().parse_args(argv), ctx)
    return rc, out.getvalue(), err.getvalue()


def _cli_job(ctx, name):
    return _job(Scheduler(ctx), name)


def test_cli_add_routine_flag_stores_true():
    ctx = _cli_ctx()
    rc, out, _ = _cli(["schedule", "add", "clijob", "every", "1h",
                       "tool", "echo", "routine"], ctx)
    assert rc == 0, out
    assert _cli_job(ctx, "clijob")["routine"] is True


def test_cli_add_routine_on_off_and_default():
    ctx = _cli_ctx()
    rc, _, _ = _cli(["schedule", "add", "clion", "every", "1h",
                     "message", "hello", "routine", "on"], ctx)
    assert rc == 0
    assert _cli_job(ctx, "clion")["routine"] is True
    rc, _, _ = _cli(["schedule", "add", "clioff", "every", "1h",
                     "message", "hello", "routine", "off"], ctx)
    assert rc == 0
    assert _cli_job(ctx, "clioff")["routine"] is False
    rc, _, _ = _cli(["schedule", "add", "cliplain", "every", "1h",
                     "message", "hello"], ctx)
    assert rc == 0
    assert _cli_job(ctx, "cliplain")["routine"] is False


# ── (b) end-to-end alerting through surface-created jobs ─────────────

def test_surface_created_routine_job_success_is_quiet_twin_alerts():
    sched = _sched(_ctx())
    chat = _chat(sched)
    chat.control("add quietone every 1h tool echo routine")
    chat.control("add loudone every 1h tool echo")
    out = sched.run_now(_job(sched, "quietone")["id"])
    assert out["ok"] is True
    assert _schedule_alerts(sched) == [], \
        "routine job created via /schedule add paged on success"
    out = sched.run_now(_job(sched, "loudone")["id"])
    assert out["ok"] is True
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1 and alerts[0]["title"].startswith("✅")


def test_surface_created_routine_job_failures_still_surface():
    # the #71 contract, re-pinned through a surface-created job: a
    # value-declared failure RECORDS as failed (it surfaces in the run
    # record) without the generic double-page; a raised (machinery)
    # failure pages the owner.
    tools = _Tools({
        "echo": _ok({"ok": False, "error": "declared boom"}),
        "explode": _err("registry blew up"),
    })
    sched = _sched(_ctx(tools))
    chat = _chat(sched)
    chat.control("add declfail every 1h tool echo routine")
    out = sched.run_now(_job(sched, "declfail")["id"])
    assert out["ok"] is False
    runs = sched.recent_runs(_job(sched, "declfail")["id"])
    assert runs[0]["ok"] is False
    assert "declared boom" in _job(sched, "declfail")["last_result"]
    assert _schedule_alerts(sched) == []
    chat.control("add raisefail every 1h tool explode routine")
    out = sched.run_now(_job(sched, "raisefail")["id"])
    assert out["ok"] is False
    alerts = _schedule_alerts(sched)
    assert len(alerts) == 1 and alerts[0]["title"].startswith("❌")


# ── (c) list/show rendering ──────────────────────────────────────────

def test_chat_list_marks_routine_only():
    sched = _sched(_ctx())
    chat = _chat(sched)
    chat.control("add marked every 1h tool echo routine")
    chat.control("add unmarked every 1h tool echo")
    listing = chat.control("list")
    lines = {ln.split("] ", 1)[-1].split(" — ")[0]: ln
             for ln in listing.splitlines() if ln.startswith("  [")}
    assert "routine" in lines["marked"]
    assert "routine" not in lines["unmarked"]


def test_cli_list_marks_routine_only():
    ctx = _cli_ctx()
    _cli(["schedule", "add", "climarked", "every", "1h",
          "message", "hi", "routine"], ctx)
    _cli(["schedule", "add", "cliunmarked", "every", "1h",
          "message", "hi"], ctx)
    rc, out, _ = _cli(["schedule", "list"], ctx)
    assert rc == 0
    lines = {ln.split("] ", 1)[-1].split(" — ")[0]: ln
             for ln in out.splitlines() if ln.startswith("  [")}
    assert "routine" in lines["climarked"]
    assert "routine" not in lines["cliunmarked"]


def test_list_jobs_json_carries_routine():
    # the formatted job dict (nm schedule list --json dumps it) names
    # the flag explicitly — pinned so the marker can't drift from data
    sched = _sched(_ctx())
    _chat(sched).control("add jsonjob every 1h tool echo routine")
    assert _job(sched, "jsonjob")["routine"] is True


# ── (d) edit toggle ──────────────────────────────────────────────────

def test_set_routine_api_toggles_stored_flag():
    sched = _sched(_ctx())
    job = sched.add("toggleme", "every 1h", "tool", {"tool": "echo"})
    assert _job(sched, "toggleme")["routine"] is False
    updated = sched.set_routine(job["id"], True)
    assert updated["routine"] is True
    assert _job(sched, "toggleme")["routine"] is True
    updated = sched.set_routine("toggleme", False)
    assert updated["routine"] is False
    with pytest.raises(LookupError):
        sched.set_routine("no such job", True)


def test_chat_routine_verb_toggles_and_next_run_follows():
    sched = _sched(_ctx())
    chat = _chat(sched)
    chat.control("add flipme every 1h tool echo")
    sched.run_now(_job(sched, "flipme")["id"])
    assert len(_schedule_alerts(sched)) == 1  # non-routine: paged
    reply = chat.control("routine flipme on")
    assert "routine on" in reply
    assert _job(sched, "flipme")["routine"] is True
    sched.run_now(_job(sched, "flipme")["id"])
    assert len(_schedule_alerts(sched)) == 1, \
        "toggled-on routine job still paged on its next success"
    reply = chat.control("routine flipme off")
    assert "routine off" in reply
    sched.run_now(_job(sched, "flipme")["id"])
    assert len(_schedule_alerts(sched)) == 2, \
        "toggled-off job did not resume paging on success"


def test_cli_routine_verb_toggles():
    ctx = _cli_ctx()
    _cli(["schedule", "add", "cliflip", "every", "1h",
          "message", "hi"], ctx)
    rc, out, _ = _cli(["schedule", "routine", "cliflip", "on"], ctx)
    assert rc == 0 and "routine on" in out
    assert _cli_job(ctx, "cliflip")["routine"] is True
    rc, out, _ = _cli(["schedule", "routine", "cliflip", "off"], ctx)
    assert rc == 0 and "routine off" in out
    assert _cli_job(ctx, "cliflip")["routine"] is False


# ── (e) invalid values rejected honestly ─────────────────────────────

def test_chat_add_routine_garbage_rejected_no_job_created():
    sched = _sched(_ctx())
    reply = _chat(sched).control(
        "add badjob every 1h tool echo routine banana")
    assert "routine" in reply and "banana" in reply
    assert "scheduled badjob" not in reply
    assert sched.list_jobs() == [], "garbage routine value created a job"


def test_chat_routine_verb_garbage_rejected_flag_unchanged():
    sched = _sched(_ctx())
    chat = _chat(sched)
    chat.control("add steady every 1h tool echo")
    reply = chat.control("routine steady banana")
    assert "usage" in reply
    assert _job(sched, "steady")["routine"] is False
    reply = chat.control("routine steady")
    assert "usage" in reply


def test_cli_add_routine_garbage_rejected_no_job_created():
    ctx = _cli_ctx()
    rc, _, err = _cli(["schedule", "add", "badcli", "every", "1h",
                       "message", "hi", "routine", "banana"], ctx)
    assert rc != 0
    assert "routine" in err
    assert Scheduler(ctx).list_jobs() == []


def test_cli_routine_verb_garbage_rejected():
    ctx = _cli_ctx()
    _cli(["schedule", "add", "clisteady", "every", "1h",
          "message", "hi"], ctx)
    rc, _, err = _cli(["schedule", "routine", "clisteady", "banana"], ctx)
    assert rc != 0
    assert _cli_job(ctx, "clisteady")["routine"] is False
