"""PASS #75: the finance alerts job actually runs — job→tool bindings pinned.

The defect (owner's EC2 jobs dump, 2026-10-21): the "finance alerts"
job failed EVERY run (every 900s) with "tool finance_alert failed".

Both sides of the mismatch:

* Job side (``agents/partner/runtime.py``, now ``finance/jobs.py``)::

      {"tool": "finance_alert",
       "args": {"action": "check", "deliver": True}}

  driven by the scheduler (``agents/scheduler.py::_run_tool``)::

      tools.call(tool, capabilities=CapabilitySet.all(), **args)

* Tool side (``finance/tools.py``)::

      def finance_alert(context: Any, action: str = "list", ...,
                        deliver: bool = False)

  The registry's dispatch (``tools/registry.py::_call_inner``) is
  ``spec.fn(**kwargs)`` — it NEVER supplies a ``context`` argument.
  The codebase's documented convention (``tools/autonomy.py``) is that
  scheduler-bound tools close over ``registry.context`` instead of
  taking it as a parameter. The finance family — 19 context-first
  tools registered through ``tools/finance.py`` — never got that
  binding, so every registry call raised::

      TypeError: finance_alert() missing 1 required positional
      argument: 'context'

  ``finance_digest`` (``def finance_digest(context)``) shared the
  defect: the Sunday digest job had never run either. Of all 27
  built-in job→tool binding instances, these two were the only ones
  that did not bind.

The fix binds the registry's context into the finance family's
functions at the ``tools/finance.py`` registration seam (the same
closure-over-``registry.context`` pattern ``pulse``/``briefing`` and
the ``__*_`` scheduler bridges already use), extracts the finance job
registrations into ``finance/jobs.py::ensure_finance_jobs`` like every
other job family, and nests proxylab's ``proxy_refresh`` job argument
under ``"args"`` (it sat at payload top level, where ``_run_tool``
never looks — the limit was silently dropped).

This suite pins the whole class: every built-in job→tool binding is
enumerated (ensure functions executed against a scratch scheduler +
the partner-runtime inline registrations) and verified against the
registered tool's real signature, and an AST guard watches the source
so a future literal registration cannot drift in unexamined.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


# ── harness ──────────────────────────────────────────────────────────

class _SchedNotifier:
    """The scheduler's own alert channel — recorded, not under test."""

    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


class _ToolNotifier:
    """Replacement for ``agents.notifier.Notifier`` — the finance
    tools' delivery path (``Notifier(context).publish(...)``)."""

    published: list = []

    def __init__(self, context):
        self.context = context

    def publish(self, kind, title, body="", **kw):
        type(self).published.append(
            {"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


def _ctx(tmp_path, *, with_tools=True):
    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={},
        tools=None,
    )
    if with_tools:
        registry = ToolRegistry(ctx)
        registry.register_builtins()
        ctx.tools = registry
    return ctx


def _sched(ctx):
    sched = Scheduler(ctx)
    sched.notifier = _SchedNotifier()
    return sched


def _job_row(sched, name):
    rows = [j for j in sched.list_jobs() if j.get("name") == name]
    assert len(rows) == 1, f"expected one job named {name!r}, got {rows}"
    return rows[0]


def _runs(ctx, job_id):
    return ctx.db.query(
        "SELECT ok, result FROM schedule_runs WHERE job_id = ?",
        (job_id,))


def _payload(ctx, name):
    """The job's stored payload, read from the table the scheduler
    itself reads (list_jobs does not expose it)."""
    row = ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?", (name,))
    assert row is not None, f"no schedule_jobs row named {name!r}"
    return json.loads(row["payload"] or "{}")


def _last_result(ctx, job_id):
    """The full stored last_result (list_jobs truncates it to 300)."""
    row = ctx.db.query_one(
        "SELECT last_result FROM schedule_jobs WHERE id = ?", (job_id,))
    assert row is not None
    return row["last_result"] or ""


def _finance_env(monkeypatch, tmp_path, price_fn):
    """Point the alerts engine at a tmp store + a stubbed price source,
    and capture the tools' Notifier. Returns the tmp AlertStore."""
    from nomorals.finance import alerts as alerts_mod

    store = alerts_mod.AlertStore(tmp_path / "alerts.json")
    monkeypatch.setattr(alerts_mod, "AlertStore", lambda path=None: store)
    monkeypatch.setattr(alerts_mod, "_default_price_fn", price_fn)
    _ToolNotifier.published = []
    monkeypatch.setattr("nomorals.agents.notifier.Notifier", _ToolNotifier)
    return store


# ── (a) the alerts job runs end-to-end through the real scheduler ────

def test_finance_alerts_job_runs_and_records_ok(tmp_path, monkeypatch):
    _finance_env(monkeypatch, tmp_path, lambda s, m: {"price": 1.0})
    ctx = _ctx(tmp_path)
    sched = _sched(ctx)
    from nomorals.finance.jobs import ensure_finance_jobs

    ensure_finance_jobs(sched)
    row = _job_row(sched, "finance alerts")
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    last = _last_result(ctx, row["id"])
    assert not last.startswith("job failed"), last
    assert last.startswith("finance_alert:"), last
    runs = _runs(ctx, row["id"])
    assert len(runs) == 1 and runs[0]["ok"] == 1, runs


def test_finance_jobs_registered_with_the_runtime_specs(tmp_path):
    ctx = _ctx(tmp_path, with_tools=False)
    sched = _sched(ctx)
    from nomorals.finance.jobs import ensure_finance_jobs

    out = ensure_finance_jobs(sched)
    assert {o["name"] for o in out} == {"finance alerts", "finance digest"}
    alerts = _job_row(sched, "finance alerts")
    assert alerts["spec"] == "every 900s", alerts["spec"]
    assert _payload(ctx, "finance alerts") == {
        "tool": "finance_alert",
        "args": {"action": "check", "deliver": True},
    }
    digest = _job_row(sched, "finance digest")
    assert digest["spec"] == "cron 0 9 * * SUN", digest["spec"]
    assert _payload(ctx, "finance digest") == {
        "tool": "finance_digest", "args": {},
    }
    # idempotent: a second registration changes nothing
    ensure_finance_jobs(sched)
    assert len([j for j in sched.list_jobs()
                if j["name"].startswith("finance ")]) == 2


# ── (b) a met condition delivers exactly one alert via the Notifier ──

def test_alert_condition_met_publishes_once_via_notifier(
        tmp_path, monkeypatch):
    store = _finance_env(
        monkeypatch, tmp_path,
        lambda s, m: {"price": 150.0, "change_pct_24h": 2.0})
    from nomorals.finance.alerts import add_alert

    add_alert("price_above", "BTC", 100.0, store=store)
    ctx = _ctx(tmp_path)
    sched = _sched(ctx)
    from nomorals.finance.jobs import ensure_finance_jobs

    ensure_finance_jobs(sched)
    out = sched.run_now(_job_row(sched, "finance alerts")["id"])
    assert out["ok"] is True, out
    assert len(_ToolNotifier.published) == 1, _ToolNotifier.published
    note = _ToolNotifier.published[0]
    assert note["kind"] == "price_alert", note
    assert "BTC" in note["title"], note
    row = _job_row(sched, "finance alerts")
    last = _last_result(ctx, row["id"])
    assert "fired: 1" in last, last


# ── (c) no condition met / no data = honest quiet success ────────────

def test_no_condition_met_is_quiet_success(tmp_path, monkeypatch):
    store = _finance_env(
        monkeypatch, tmp_path,
        lambda s, m: {"price": 50.0, "change_pct_24h": 0.5})
    from nomorals.finance.alerts import add_alert

    add_alert("price_above", "BTC", 100.0, store=store)
    ctx = _ctx(tmp_path)
    sched = _sched(ctx)
    from nomorals.finance.jobs import ensure_finance_jobs

    ensure_finance_jobs(sched)
    out = sched.run_now(_job_row(sched, "finance alerts")["id"])
    assert out["ok"] is True, out
    assert _ToolNotifier.published == []
    row = _job_row(sched, "finance alerts")
    last = _last_result(ctx, row["id"])
    assert "fired: 0" in last, last


def test_price_source_unavailable_is_quiet_success(tmp_path, monkeypatch):
    # The alerts engine's documented contract: a dead source is
    # "no data", not a crash — the run records a clean no-fire success.
    store = _finance_env(monkeypatch, tmp_path, lambda s, m: None)
    from nomorals.finance.alerts import add_alert

    add_alert("price_above", "BTC", 100.0, store=store)
    ctx = _ctx(tmp_path)
    sched = _sched(ctx)
    from nomorals.finance.jobs import ensure_finance_jobs

    ensure_finance_jobs(sched)
    out = sched.run_now(_job_row(sched, "finance alerts")["id"])
    assert out["ok"] is True, out
    assert _ToolNotifier.published == []


# ── (e) the digest job shared the defect — it runs now too ───────────

def test_finance_digest_job_runs_and_publishes(tmp_path, monkeypatch):
    _ToolNotifier.published = []
    monkeypatch.setattr("nomorals.agents.notifier.Notifier", _ToolNotifier)
    ctx = _ctx(tmp_path)
    sched = _sched(ctx)
    from nomorals.finance.jobs import ensure_finance_jobs

    ensure_finance_jobs(sched)
    row = _job_row(sched, "finance digest")
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    last = _last_result(ctx, row["id"])
    assert not last.startswith("job failed"), last
    kinds = [p["kind"] for p in _ToolNotifier.published]
    assert kinds == ["finance_digest"], _ToolNotifier.published


# ── (d) the binding contract over every built-in job ─────────────────

#: The partner runtime's inline tool-job registrations (its boot block
#: has no ensure function): job name → (tool, args). Everything else
#: built-in registers through an ensure_* function and is collected by
#: executing them, below.
INLINE_RUNTIME_BINDINGS = {
    "improvement tick": ("improve", {"action": "cycle"}),
    "mining session": ("train_mine", {}),
    "research organ tick": ("research_organ_tick", {}),
    "wisdom organ tick": ("wisdom_organ_tick", {}),
    "memory consolidation (organ loop)": ("memory_consolidate", {}),
    "cognitive loop": ("autonomy", {"action": "tick"}),
    "monitor tick": ("monitor", {"action": "tick"}),
}


def _collect_ensure_bindings(ctx, sched):
    """Execute every built-in ensure_* job registration against the
    scratch scheduler; return ({(tool, args-json): job name}, errors)."""
    from nomorals.agency.wiring import ensure_agency_shift_job
    from nomorals.agents.modes import ensure_agent_modes_tick_job
    from nomorals.agents.morning_briefing import ensure_briefing_job
    from nomorals.agents.morning_pulse import (
        ensure_pulse_job, ensure_pulse_jobs)
    from nomorals.agents.news_jobs import ensure_news_jobs
    from nomorals.agents.afternoon_pulse import ensure_afternoon_pulse_job
    from nomorals.agents.evening_pulse import ensure_evening_pulse_job
    from nomorals.agents.research_loop import ensure_research_job
    from nomorals.agents.scheduler import ensure_persona_jobs
    from nomorals.agents.skill_evolution import ensure_improvement_schedule
    from nomorals.agents.watchers import ensure_sweeper_job
    from nomorals.autonomy.presence import ensure_heartbeat_job
    from nomorals.finance.jobs import ensure_finance_jobs
    from nomorals.immune.wiring import ensure_immune_sampler_job
    from nomorals.memory.cadence import ensure_consolidation_job
    from nomorals.workspace.rooms import ensure_rooms_tick_job

    calls = [
        ("agency shift", lambda: ensure_agency_shift_job(sched)),
        ("agent modes tick", lambda: ensure_agent_modes_tick_job(sched)),
        ("immune sampler", lambda: ensure_immune_sampler_job(sched)),
        ("rooms tick", lambda: ensure_rooms_tick_job(ctx)),
        ("watchers sweep", lambda: ensure_sweeper_job(ctx)),
        ("memory cadence", lambda: ensure_consolidation_job(
            ctx, scheduler_factory=Scheduler)),
        ("persona jobs", lambda: ensure_persona_jobs(ctx)),
        ("presence heartbeat", lambda: ensure_heartbeat_job(sched)),
        ("briefing", lambda: ensure_briefing_job(ctx)),
        ("pulse jobs", lambda: ensure_pulse_jobs(ctx)),
        ("pulse (legacy single)", lambda: ensure_pulse_job(ctx)),
        ("afternoon pulse", lambda: ensure_afternoon_pulse_job(ctx)),
        ("evening pulse", lambda: ensure_evening_pulse_job(ctx)),
        ("research loop", lambda: ensure_research_job(ctx)),
        ("improvement schedule", lambda: ensure_improvement_schedule(ctx)),
        ("finance jobs", lambda: ensure_finance_jobs(sched)),
        ("news jobs", lambda: ensure_news_jobs(sched)),
    ]
    errors = {}
    for label, fn in calls:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - collected, asserted below
            errors[label] = f"{type(exc).__name__}: {exc}"
    bindings = {}
    for row in ctx.db.query(
            "SELECT name, payload_kind, payload FROM schedule_jobs"):
        if row.get("payload_kind") != "tool":
            continue
        try:
            payload = json.loads(row.get("payload") or "{}")
        except ValueError:
            continue
        tool = payload.get("tool")
        if not tool:
            continue
        args = payload.get("args") or {}
        bindings[(tool, json.dumps(args, sort_keys=True))] = row["name"]
    return bindings, errors


def test_every_builtin_job_tool_binding_binds(tmp_path):
    ctx = _ctx(tmp_path, with_tools=False)
    sched = _sched(ctx)
    bindings, errors = _collect_ensure_bindings(ctx, sched)
    assert errors == {}, f"ensure registrations failed: {errors}"
    for job_name, (tool, args) in INLINE_RUNTIME_BINDINGS.items():
        bindings[(tool, json.dumps(args, sort_keys=True))] = job_name
    # coverage sanity: the enumeration must actually see the families
    tools_seen = {tool for tool, _ in bindings}
    for expected in ("finance_alert", "finance_digest", "pulse", "briefing",
                     "memory", "room", "watch", "research_loop", "improve",
                     "skill_evolve", "skill_synthesize",
                     "__presence_heartbeat__", "__agency_shift__",
                     "__agent_modes_tick__", "__immune_sampler__"):
        assert expected in tools_seen, \
            f"enumeration missed {expected!r}; saw {sorted(tools_seen)}"
    assert len(bindings) >= 24, f"only {len(bindings)} bindings enumerated"

    registry = ToolRegistry(SimpleNamespace(db=None, settings=None))
    registry.register_builtins()
    failures = []
    for (tool, args_json), job_name in sorted(bindings.items()):
        args = json.loads(args_json)
        spec = registry._tools.get(tool)
        if spec is None:
            failures.append(f"{job_name}: tool {tool!r} not in registry")
            continue
        try:
            # The scheduler's exact call shape: registry dispatch calls
            # spec.fn(**args) — bind exactly that against the real
            # (post-binding) signature.
            inspect.signature(spec.fn).bind(**args)
        except Exception as exc:  # noqa: BLE001 - collected per binding
            failures.append(f"{job_name} → {tool}{args}: {exc}")
    assert failures == [], "unbindable job→tool bindings:\n" + "\n".join(
        failures)


#: Files whose scheduler.add payload names its tool through a variable
#: (a module constant or a loop variable) — the AST guard below cannot
#: read those statically; they are covered by executing their ensure
#: functions / the inline table above. A NEW file appearing here fails
#: the guard until a human classifies its registration.
_VARIABLE_TOOL_FILES = {
    "agency/wiring.py",          # {"tool": AGENCY_SHIFT_TOOL}
    "agents/modes.py",           # {"tool": AGENT_MODES_TICK_TOOL}
    "agents/skill_evolution.py",  # loop over (name, spec, tool, args)
    "immune/wiring.py",          # {"tool": IMMUNE_SAMPLER_TOOL}
    "agents/partner/runtime.py",  # organ-tick loop {"tool": _tool}
}


def _literal_tool_payloads():
    """AST-scan nomorals/ for scheduler.add(...) calls whose payload
    dict literal carries a "tool" key. Yields (relpath, tool, keys)."""
    root = Path(__file__).resolve().parent.parent / "nomorals"
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add"):
                continue
            dicts = [a for a in node.args if isinstance(a, ast.Dict)]
            dicts += [k.value for k in node.keywords
                      if isinstance(k.value, ast.Dict)]
            for d in dicts:
                keys = {}
                for k, v in zip(d.keys, d.values):
                    if isinstance(k, ast.Constant) \
                            and isinstance(k.value, str):
                        keys[k.value] = v
                if "tool" not in keys:
                    continue
                tool_v = keys["tool"]
                tool = tool_v.value if isinstance(tool_v, ast.Constant) \
                    else None
                found.append((rel, tool, sorted(keys)))
    return found


def test_literal_job_payloads_are_wellformed_and_covered(tmp_path):
    ctx = _ctx(tmp_path, with_tools=False)
    sched = _sched(ctx)
    bindings, errors = _collect_ensure_bindings(ctx, sched)
    assert errors == {}
    covered_tools = {tool for tool, _ in bindings}
    covered_tools |= {tool for tool, _ in INLINE_RUNTIME_BINDINGS.values()}
    covered_tools.add("proxy_refresh")  # created by the proxy_schedule
    # tool, verified end-to-end in test_proxy_schedule_job_payload below

    problems = []
    variable_files = set()
    for rel, tool, keys in _literal_tool_payloads():
        if set(keys) - {"tool", "args"}:
            problems.append(
                f"{rel}: payload keys {keys} — arguments outside 'args' "
                "are silently dropped by the scheduler's _run_tool")
        if tool is None:
            variable_files.add(rel)
        elif tool not in covered_tools:
            problems.append(
                f"{rel}: job binds tool {tool!r}, which the binding "
                "contract does not enumerate — classify it in "
                "test_pass75_finance_alerts.py")
    assert problems == [], "\n".join(problems)
    assert variable_files == _VARIABLE_TOOL_FILES, (
        f"variable-tool registration files changed: "
        f"{sorted(variable_files)} != {sorted(_VARIABLE_TOOL_FILES)} — "
        "classify the new registration in the contract enumeration")


def test_proxy_schedule_job_payload_nests_its_args(tmp_path):
    # The proxy_schedule TOOL creates the proxy_refresh job; its limit
    # must sit under "args" where _run_tool looks for it.
    ctx = _ctx(tmp_path)
    registry = ctx.tools
    outcome = registry.call("proxy_schedule", enabled="true",
                            every_minutes="30")
    assert outcome.ok, getattr(outcome.error, "message", outcome.error)
    payload = _payload(ctx, "proxy_refresh")
    assert payload == {"tool": "proxy_refresh",
                       "args": {"limit": "150"}}, payload
