"""Tail-103 pins: nomorals/agents/role_specs.py census sites.

All three sites are held as exclusions:

138 + 150 — ``record_tool_denial``'s two telemetry sinks
(event-bus emit, then the failures-ledger INSERT). The
function's contract is printed on it: "Best-effort denial
telemetry... Never raises; enforcement must not depend on
telemetry succeeding." The denial itself is decided and
returned by ``check_tool_call`` before this recorder runs;
these rows are the annotation beside that decision, and
each sink is independent — a dead bus still leaves the
ledger row, a dead ledger still leaves the bus event
(pinned below). Observability only: shelved sketch is to
bind both handlers to a debug log so a permanently dead
sink is at least visible in logs.

415 — ``RoleRegistry.from_settings``: reading
``extra_role_files`` off a duck-typed settings object
raising -> no extra paths; the registry still builds
with every built-in role. Settings objects are foreign
objects by design here, and the failure degrades to the
built-in set — roles never fail open because of it
(unknown roles still resolve to the safe execution spec).

NOT-recording is pinned with the snapshot method for the
in-process ledgers: the emit/INSERT sinks are the
recording under test, and neither agency nor agents
ledger gains a row from these triggers.
"""
from __future__ import annotations

from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import role_specs as R


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


class _FakeDB:
    def __init__(self, fail: bool = False) -> None:
        self.rows: list = []
        self.fail = fail

    def execute(self, sql, params=()):
        if self.fail:
            raise RuntimeError("db boom")
        self.rows.append((sql, params))


# ── 138: a dead event bus never sinks the denial record ────────

def test_138_emit_failure_still_writes_ledger_row():
    before = _snap()
    db = _FakeDB()
    ctx = SimpleNamespace(emit=_raise, db=db)
    assert R.record_tool_denial(
        ctx, "scout", "shell", "path_guard", "blocked") is None
    # the independent ledger sink still recorded the denial
    assert len(db.rows) == 1
    assert db.rows[0][1][4] == "tool_denied"
    # control: a healthy bus receives the event
    seen = []
    db2 = _FakeDB()
    ctx2 = SimpleNamespace(
        emit=lambda event, **kw: seen.append((event, kw)), db=db2)
    R.record_tool_denial(ctx2, "scout", "shell", "path_guard", "blocked")
    assert seen == [("swarm.tool_denied",
                     {"role": "scout", "tool": "shell",
                      "reason": "path_guard"})]
    assert len(db2.rows) == 1
    assert _snap() == before


# ── 150: a dead ledger never raises out of the recorder ────────

def test_150_ledger_failure_is_swallowed():
    before = _snap()
    seen = []
    ctx = SimpleNamespace(
        emit=lambda event, **kw: seen.append(event),
        db=_FakeDB(fail=True))
    assert R.record_tool_denial(
        ctx, "scout", "shell", "read_only", "blocked") is None
    # the bus sink still fired — the sinks are independent
    assert seen == ["swarm.tool_denied"]
    assert _snap() == before


# ── 415: raising settings still build the built-in registry ────

class _BadSettings:
    @property
    def extra_role_files(self):
        raise RuntimeError("settings boom")


def test_415_raising_settings_fall_back_to_builtins():
    before = _snap()
    registry = R.RoleRegistry.from_settings(_BadSettings())
    # built-ins are all there; unknown roles stay fail-closed
    assert registry.resolve("execution").name == "execution"
    assert registry.resolve("no-such-role").name == "execution"
    # control: a well-behaved settings object takes the same path
    registry2 = R.RoleRegistry.from_settings(
        SimpleNamespace(extra_role_files=[]))
    assert registry2.resolve("execution").name == "execution"
    assert _snap() == before
