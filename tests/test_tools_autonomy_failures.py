"""Tail-81 failure pins — nomorals/tools/autonomy.py.

All six sites are EXCLUSIONS, probed live first
(nm-tail81-verify/probes.py, 24/24 behavior confirmations):

- 94: weakness_dismiss guards db.commit() — a vestigial call
  on the production storage Database, which has NO commit
  method (AttributeError every call) and connects autocommit,
  so the UPDATE persists at execute() time. The returned
  {"ok": bool(cur.rowcount)} is the truthful outcome
  (tail-67 vestigial-call family).
- 164: _gradient_for_shift swallows an unreadable rules
  table into a rule-less Policy — default-deny. The failure
  NARROWS the scheduled shift's authority, never widens it;
  the module states the contract in-line.
- 230: the agency shift's _try ladder turns a failed organ
  construction into None, which build_adapters treats as an
  organ-less adapter (empty observations, refused actions);
  the agency layer records honest "no organ injected"
  blockers. Fail-closed by design.
- 303: the DM sender returns False when gateway.send raises
  — the honest "not sent" value; True is returned only when
  the send result's .ok is true.
- 354: a failing context.extras lookup falls back to a
  default AgentModes engine, which still ticks chats from
  the settings-resolved store (documented fallback rung).
- 431: immune-sampler reporting is best-effort — a failing
  Notifier.publish never breaks the tick, and the tick result
  still returns to the scheduler, which records it as the
  job's last_result and in the run ledger (evidence kept;
  silence is a chat-surface rule only).

The agency-shift pins capture build_adapters' kwargs via a
sentinel: every _try() runs and the dm sender is built while
the tool evaluates that call's arguments, so the capture
holds the exact artifacts of sites 230/303 without running
the (network-bound) shift itself.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

import nomorals.agency.adapters as adapters_mod
from nomorals.autonomy.weakness import ensure_schema as w_schema
from nomorals.immune import wiring
from nomorals.storage.db import Database
from nomorals.tools import autonomy as ta


class FakeRegistry:
    """Registry stand-in: register() supports both the plain
    call and the decorator style used by register()."""

    def __init__(self, context):
        self.context = context
        self.fns = {}

    def register(self, name, fn=None, **kw):
        if fn is None:
            def deco(f):
                self.fns[name] = f
                return f
            return deco
        self.fns[name] = fn
        return fn


class _Sentinel(Exception):
    pass


def _capture_shift_kwargs(context):
    captured = {}

    def capturing_build(**kw):
        captured.update(kw)
        raise _Sentinel()

    reg = FakeRegistry(context)
    ta.register(reg)
    with mock.patch.object(adapters_mod, "build_adapters",
                           capturing_build):
        with pytest.raises(_Sentinel):
            reg.fns["__agency_shift__"]()
    return captured


def test_94_weakness_dismiss_commit_guard_is_vestigial_on_autocommit(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    db = Database(ws / "nomorals.db")
    w_schema(db)
    db.execute("INSERT INTO weaknesses (id, kind, subject, status)"
               " VALUES ('w1', 'probe', 'probe weakness', 'proposed')")
    # The guarded call can only ever raise: the production
    # Database has no commit method. Autocommit has already
    # persisted the UPDATE before the guard runs.
    assert not hasattr(db, "commit")
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(ws)))
    with mock.patch.object(
            ta, "_db", lambda c: Database(ws / "nomorals.db")):
        assert ta.weakness_dismiss(ctx, "w1") == {"ok": True}
        assert ta.weakness_dismiss(ctx, "no-such") == {"ok": False}
    fresh = Database(ws / "nomorals.db")
    row = fresh.query_one(
        "SELECT status FROM weaknesses WHERE id = 'w1'")
    assert row["status"] == "dismissed"


def test_164_gradient_unreadable_rules_stays_default_deny():
    class RaiseDB:
        def execute(self, *a, **k):
            raise RuntimeError("schema boom")

        def query(self, *a, **k):
            raise RuntimeError("query boom")

    grad = ta._gradient_for_shift(RaiseDB())
    # No rules loaded = default-deny stands; the failure
    # narrowed authority instead of widening it.
    assert grad.policy.rules_to_dict() == []


def test_230_agency_shift_organ_construction_failure_yields_none(tmp_path):
    ctx = SimpleNamespace(
        settings=SimpleNamespace(
            workspace_dir=str(tmp_path / "shiftws")),
        gateway=None, extras={})
    healthy = _capture_shift_kwargs(ctx)
    assert healthy["news_agent"] is not None
    with mock.patch("nomorals.agents.news.NewsAgent",
                    side_effect=RuntimeError("news down")):
        broken = _capture_shift_kwargs(ctx)
    # The failed construction degrades to the organ-less
    # adapter value; every other organ is unaffected.
    assert broken["news_agent"] is None
    assert broken["weather"] is not None


class _FakeGateway:
    def __init__(self, mode):
        self.mode = mode

    def send(self, platform, contact, body):
        if self.mode == "raise":
            raise RuntimeError("send boom")
        return SimpleNamespace(ok=self.mode == "ok")


def test_303_dm_sender_send_failure_is_not_sent(tmp_path):
    def sender_for(mode):
        gateway = None if mode is None else _FakeGateway(mode)
        ctx = SimpleNamespace(
            settings=SimpleNamespace(
                workspace_dir=str(tmp_path / "ws")),
            gateway=gateway, extras={})
        return _capture_shift_kwargs(ctx)["dm_sender"]

    assert sender_for(None) is None
    # A raising gateway yields the honest "not sent" value.
    assert sender_for("raise")("tg", "c", "hi") is False
    assert sender_for("notok")("tg", "c", "hi") is False
    assert sender_for("ok")("tg", "c", "hi") is True


def test_354_agent_modes_extras_failure_falls_back_to_default_engine():
    class RaisingExtrasCtx:
        settings = SimpleNamespace(workspace_dir=".")
        gateway = None

        @property
        def extras(self):
            raise RuntimeError("extras boom")

    reg = FakeRegistry(RaisingExtrasCtx())
    ta.register(reg)
    out = reg.fns["__agent_modes_tick__"]()
    # The fallback engine ran (no crash, no "no agent modes"
    # error): a normal tick summary comes back.
    assert out["ok"] is True
    assert isinstance(out["chats"], list)
    assert "error" not in out

    # Control: a readable extras engine is the one ticked.
    fake_modes = SimpleNamespace(
        tick_all=lambda cats: {"chatA": []})
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="."),
        gateway=None, extras={"agent_modes": fake_modes})
    reg2 = FakeRegistry(ctx)
    ta.register(reg2)
    out2 = reg2.fns["__agent_modes_tick__"]()
    assert out2["ok"] is True and out2["chats"] == ["chatA"]


def test_431_immune_reporting_failure_never_breaks_the_tick(tmp_path):
    class FakeMonitor:
        def tick(self):
            return {"ok": True, "sampled": True}

        def drain_announcements(self):
            return []

    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="."),
        gateway=None, extras={})
    reg = FakeRegistry(ctx)
    ta.register(reg)
    event = {"title": "t", "critical": False,
             "subject": "s", "hints": {}}
    with mock.patch.object(wiring, "default_store_path",
                           lambda c: tmp_path / "imm"), \
            mock.patch.object(wiring, "shared_monitor",
                              lambda p: FakeMonitor()), \
            mock.patch.object(wiring, "summarize_tick_event",
                              lambda result, ann: event), \
            mock.patch("nomorals.agents.notifier.Notifier.publish",
                       side_effect=RuntimeError("publish boom")):
        out = reg.fns["__immune_sampler__"]()
    # Reporting failed; the tick result is returned verbatim
    # (and is what the scheduler records as last_result).
    assert out == {"ok": True, "sampled": True}
