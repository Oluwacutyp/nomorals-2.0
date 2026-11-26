"""C.10 (§3C.10): crontab parser + next-event solver — echelon/crontab.rs
+ task.rs pattern (BSD-4).

Devon's cron engine (nomorals/scheduler/recurrence.py) already covered
the donor and went past it — except month names: ``0 9 * JAN MON``
failed with a raw ``invalid literal for int()`` while day names worked.
This lane adds month names (abbreviations + full names, lists/ranges/
steps) to both cron parsers, and makes the next-event solver
exercisable without creating a job: ``/schedule cron`` and
``nm schedule cron`` validate + describe + preview an expression.

No new brain tool: the schedule surface already exists, this is a verb
on it — the registry stays at 787.
"""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nomorals.agents.partner.runtime_schedule import RuntimeScheduleMixin
from nomorals.agents.scheduler import Scheduler
from nomorals.scheduler.recurrence import (
    CronNeverMatchesError,
    CronTimezoneError,
    describe_cron,
    inspect_cron,
    next_cron,
    next_n_cron,
    parse_cron,
    parse_cron_preview_args,
    render_cron_preview,
)
from nomorals.scheduler.scheduler import CronParser
from nomorals.storage.db import Database

# ── month names: CronSpec ────────────────────────────────────────────────

def test_month_names_parse():
    spec = parse_cron("0 9 * JAN MON")
    assert spec.month == {1}
    assert spec.expression == "0 9 * JAN MON"  # name kept, like MON
    assert parse_cron("0 9 * JAN-MAR *").month == {1, 2, 3}
    assert parse_cron("0 9 * JAN,APR,JUL *").month == {1, 4, 7}
    assert parse_cron("0 9 * JAN-DEC/3 *").month == {1, 4, 7, 10}
    assert parse_cron("0 9 * jan mon").expression == "0 9 * JAN MON"
    assert parse_cron("0 9 * JANUARY *").month == {1}
    assert parse_cron("0 9 * SEPTEMBER *").month == {9}
    assert parse_cron("0 9 * SEPT *").month == {9}
    # day names keep working alongside month names
    assert parse_cron("0 9 * JAN MON-FRI").dow.days == {1, 2, 3, 4, 5}


def test_month_names_next_cron():
    # 2027-01-04 is a Monday; the first January Monday after Nov 2026.
    nxt = next_cron("0 9 * JAN MON", datetime(2026, 11, 20, 10, 0))
    assert datetime.fromtimestamp(nxt) == datetime(2027, 1, 4, 9, 0)
    nxt = next_cron("0 9 1 JAN *", datetime(2026, 11, 20, 10, 0))
    assert datetime.fromtimestamp(nxt) == datetime(2027, 1, 1, 9, 0)
    # months stay restricted: no December firing for a JAN expression
    nxt = next_cron("0 9 * JAN *", datetime(2026, 12, 15, 10, 0))
    assert datetime.fromtimestamp(nxt).month == 1


def test_month_names_describe():
    text = describe_cron("0 9 * JAN MON")
    assert "January" in text and "Monday" in text
    assert "9:00 AM" in text


def test_month_names_reject_by_name():
    with pytest.raises(ValueError, match="bad cron value 'FOO'"):
        parse_cron("0 9 * FOO *")
    with pytest.raises(ValueError, match="out of range"):
        parse_cron("0 9 * JAN-13 *")
    with pytest.raises(ValueError, match="out of range"):
        parse_cron("0 9 * 13 *")


def test_step_zero_refused_not_a_crash():
    # the donor's step math divides by the step; a 0 step must be a
    # named refusal, never a ZeroDivisionError.
    with pytest.raises(ValueError, match="bad cron step"):
        parse_cron("*/0 * * * *")
    with pytest.raises(ValueError, match="bad cron step"):
        parse_cron("0 9 * JAN-DEC/0 *")


def test_vixie_step_semantics_unchanged():
    # Devon deliberately anchors steps at the range start (Vixie), NOT
    # at 0 like the donor (1-20/5 → 5,10,15,20 there). Pinned: do not
    # "fix" this to the donor's reading — it would move every job.
    assert parse_cron("1-20/5 * * * *").minute == {1, 6, 11, 16}
    assert parse_cron("2/5 * * * *").minute == {2, 7, 12, 17, 22, 27,
                                                32, 37, 42, 47, 52, 57}


# ── next_n_cron ──────────────────────────────────────────────────────────

def test_next_n_cron():
    firings = next_n_cron("0 9 * JAN MON", datetime(2026, 11, 20, 10, 0), 3)
    assert len(firings) == 3
    assert firings == sorted(firings)
    assert all(f > datetime(2026, 11, 20, 10, 0).timestamp()
               for f in firings)
    assert [datetime.fromtimestamp(f) for f in firings] == [
        datetime(2027, 1, 4, 9, 0),
        datetime(2027, 1, 11, 9, 0),
        datetime(2027, 1, 18, 9, 0),
    ]
    assert next_n_cron("0 9 * * *", datetime(2026, 11, 20, 10, 0), 0) == []
    with pytest.raises(ValueError, match="caps at 100"):
        next_n_cron("0 9 * * *", datetime(2026, 11, 20, 10, 0), 101)


def test_next_n_cron_refusals():
    with pytest.raises(CronNeverMatchesError, match="can never match"):
        next_n_cron("0 0 31 2 *", datetime(2026, 1, 1), 2)
    with pytest.raises(CronTimezoneError, match="unknown timezone"):
        next_n_cron("0 9 * * *", datetime(2026, 1, 1), 2,
                    timezone="No/Such_Zone")


def test_next_n_cron_timezone():
    zone = ZoneInfo("America/New_York")
    firings = next_n_cron("0 9 * * *", datetime(2026, 11, 20, 10, 0),
                          2, timezone="America/New_York")
    assert len(firings) == 2
    for ts in firings:
        local = datetime.fromtimestamp(ts, tz=zone)
        assert (local.hour, local.minute) == (9, 0)


# ── CronParser (triggers + user-facing Scheduler path) ───────────────────

def test_cron_parser_month_names():
    assert CronParser.parse("0 9 * JAN-MAR *")["month"] == {1, 2, 3}
    assert CronParser.parse("0 9 * JAN *")["month"] == {1}
    assert CronParser.parse("0 9 * january *")["month"] == {1}
    assert CronParser.MONTH_NAMES["JAN"] == 1
    assert CronParser.MONTH_NAMES["DECEMBER"] == 12
    # day names still resolve through the same parser
    assert CronParser.parse("0 9 * * MON-FRI")["day_of_week"] == {1, 2, 3, 4, 5}


# ── inspect / preview args / render ──────────────────────────────────────

def test_inspect_cron():
    result = inspect_cron("0 9 * JAN MON")
    assert result["expression"] == "0 9 * JAN MON"
    assert "January" in result["description"]
    assert result["timezone"] == ""
    assert len(result["next"]) == 3
    assert result["next"] == sorted(result["next"])
    result = inspect_cron("0 9 * * *", n=1, timezone="Africa/Lagos")
    assert result["timezone"] == "Africa/Lagos"
    assert len(result["next"]) == 1


def test_parse_cron_preview_args():
    assert parse_cron_preview_args(
        ["0", "9", "*", "JAN", "MON"]) == ("0 9 * JAN MON", "", 3)
    assert parse_cron_preview_args(
        ["0", "9", "*", "*", "*", "tz", "Africa/Lagos", "n", "2"]
    ) == ("0 9 * * *", "Africa/Lagos", 2)
    assert parse_cron_preview_args(
        ["0", "9", "*", "*", "*", "--tz", "UTC", "--n", "1"]
    ) == ("0 9 * * *", "UTC", 1)
    with pytest.raises(ValueError, match="5-field"):
        parse_cron_preview_args(["0", "9", "*"])
    with pytest.raises(ValueError, match="unknown option"):
        parse_cron_preview_args(["0", "9", "*", "*", "*", "bogus", "1"])
    with pytest.raises(ValueError, match="n must be 1..10"):
        parse_cron_preview_args(["0", "9", "*", "*", "*", "n", "99"])


def test_render_cron_preview():
    result = inspect_cron("0 9 * JAN MON", n=2)
    text = render_cron_preview(result)
    assert text.splitlines()[0] == "cron 0 9 * JAN MON"
    assert "January" in text
    assert "2027-01-04 09:00" in text
    assert "2027-01-11 09:00" in text


# ── chat verb: /schedule cron ────────────────────────────────────────────

class _Tools:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    def call(self, tool, capabilities=None, **args):
        outcome = self._outcomes[tool]
        return outcome(**args) if callable(outcome) else outcome


def _ok(value):
    return SimpleNamespace(ok=True, value=value, error=None)


def _ctx():
    db = Database(":memory:")
    db.migrate()
    tools = _Tools({"echo": _ok({"ok": True, "said": "hi"})})
    return SimpleNamespace(db=db, tools=tools,
                           settings=SimpleNamespace(), extras={})


class _Notifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


def _sched(ctx):
    sched = Scheduler(ctx)
    sched.notifier = _Notifier()
    return sched


class _ChatStub(RuntimeScheduleMixin):
    """Smallest PartnerRuntime stand-in: a preset _scheduler."""

    def __init__(self, scheduler):
        self._scheduler = scheduler
        self.context = scheduler.context
        self.gateway = None

    def control(self, tail):
        return self._control_schedule(tail)


def test_chat_cron_inspects():
    reply = _ChatStub(_sched(_ctx())).control("cron 0 9 * JAN MON")
    assert "cron 0 9 * JAN MON" in reply
    assert "At 9:00 AM" in reply and "January" in reply
    assert "2027-01-04 09:00" in reply


def test_chat_cron_refusals_name_the_problem():
    chat = _ChatStub(_sched(_ctx()))
    assert "can never match" in chat.control("cron 0 0 31 2 *")
    assert "bad cron value 'FOO'" in chat.control("cron 0 9 * FOO *")
    assert "unknown timezone" in chat.control(
        "cron 0 9 * * * tz No/Such_Zone")
    assert "usage:" in chat.control("cron")


def test_chat_cron_creates_month_name_job():
    sched = _sched(_ctx())
    chat = _ChatStub(sched)
    reply = chat.control(
        "add jan9 0 9 * JAN * message january nine")
    assert "scheduled jan9" in reply
    jobs = [j for j in sched.list_jobs() if j.get("name") == "jan9"]
    assert len(jobs) == 1
    nxt = datetime.fromtimestamp(jobs[0]["next_run"])
    assert nxt.month == 1 and (nxt.hour, nxt.minute) == (9, 0)


# ── CLI verb: nm schedule cron ───────────────────────────────────────────

def test_cli_cron_prose():
    from nomorals.cmdline.dispatch import main

    out = io.StringIO()
    with redirect_stdout(out):
        rc = main(["schedule", "cron", "0", "9", "*", "JAN", "MON"])
    assert rc == 0
    text = out.getvalue()
    assert "cron 0 9 * JAN MON" in text
    assert "January" in text and "2027-01-04 09:00" in text


def test_cli_cron_json():
    from nomorals.cmdline.dispatch import main

    out = io.StringIO()
    with redirect_stdout(out):
        rc = main(["schedule", "cron", "0", "9", "*", "JAN", "MON",
                   "n", "2", "--json"])
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["expression"] == "0 9 * JAN MON"
    assert "January" in payload["description"]
    assert len(payload["next"]) == 2


def test_cli_cron_refuses():
    from nomorals.cmdline.dispatch import main

    err = io.StringIO()
    with redirect_stderr(err):
        rc = main(["schedule", "cron", "0", "0", "31", "2", "*"])
    assert rc == 1
    assert "can never match" in err.getvalue()


# ── registry pin: no new tool ────────────────────────────────────────────

def test_tool_count_still_804():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 804
