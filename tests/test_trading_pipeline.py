"""Pass 6 of Devon's trading system: brain control + autonomy.

The pipeline (agents/trade_pipeline.py) drives Passes 1–5 on the
durable Scheduler (finance/trading_jobs.py registers two tool
payload jobs — registration survives restarts, idempotent by
name), folds its control state from the Pass 1 journal, honors
the Pass 4 HALT kill switch absolutely, and is slot-idempotent:
a doubled tick journals once and acts once. The brain surface is
the trading tool's pipeline_* actions; the CLI mirrors it.

Every test here is a NEW-BEHAVIOR test: on pristine de02507 the
pipeline does not exist and the file fails at import. Pass 1–5
behavior is carried by their own suites + neighborhoods.
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from nomorals.agents import forecast as fc
from nomorals.agents import forecast_trained as ft
from nomorals.agents import trade_pipeline as pipeline
from nomorals.finance import trading_journal
from nomorals.integrations import market_data

BASE = 1762362000.0  # 2025-11-05 17:00 UTC
_STEP = {"1d": 86400, "4h": 14400, "1h": 3600,
         "30m": 1800, "15m": 900, "5m": 300}


# ── fakes ────────────────────────────────────────────────────────────

class FakeEngine:
    """The Pass 4 engine's pipeline-facing surface."""

    def __init__(self, *, desk=True, rung="paper"):
        self.desk = object() if desk else None
        self.rung = rung
        self.manage_calls: list = []

    def manage_pass(self, *, symbol=None):
        self.manage_calls.append(symbol)
        return {"kind": "manage_pass", "rung": self.rung,
                "counts": {"positions": 0, "actions": 0}}


def _cfg(**kw):
    base = {"symbols": ["XAUUSD"], "market": "forex", "mode": "paper",
            "tick_spec": "every 15m", "audit_spec": "every 6h",
            "bucket_s": 900}
    base.update(kw)
    return pipeline.PipelineConfig(**base)


@pytest.fixture
def stages(monkeypatch):
    """Stage fakes at the real module seams — call counts prove
    what a tick did and did not run."""
    calls = {"read": 0, "shadow": 0, "judge": 0, "forecast": 0,
             "resolve": 0}

    def fake_read(symbol, market="crypto", **kw):
        calls["read"] += 1
        return {"symbol": symbol, "taken_at": kw.get("now")}

    def fake_shadow(symbol, market="crypto", **kw):
        calls["shadow"] += 1
        return {"created": [], "armed": [], "triggered": [],
                "invalidated": [], "expired": [],
                "detector_errors": []}

    def fake_judge(symbol, market="crypto", **kw):
        calls["judge"] += 1
        return {"counts": {"judged": 0, "act": 0, "abstain": 0}}

    def fake_forecasts(symbol, market="crypto", **kw):
        calls["forecast"] += 1
        return [{"horizon": tag, "method": fc.METHOD,
                 "direction": "NEUTRAL"} for tag in fc.HORIZONS]

    def fake_resolve(symbol, market="crypto", **kw):
        calls["resolve"] += 1
        return 0

    monkeypatch.setattr(pipeline.market_read, "whole_market_read",
                        fake_read)
    monkeypatch.setattr(pipeline.setups_mod, "shadow_pass",
                        fake_shadow)
    monkeypatch.setattr(pipeline.judgment_mod, "judgment_pass",
                        fake_judge)
    monkeypatch.setattr(fc, "make_forecasts", fake_forecasts)
    monkeypatch.setattr(fc, "resolve_due_forecasts", fake_resolve)
    monkeypatch.setattr(ft, "latest_model",
                        lambda *a, **kw: None)
    return calls


def _ticks(journal):
    return trading_journal.read_events(limit=0, path=journal,
                                       event="pipeline_tick")


# ── config + control state ───────────────────────────────────────────

def test_config_defaults_and_settings_override():
    cfg = pipeline.pipeline_config(None)
    assert cfg.symbols == ["XAUUSD"] and cfg.market == "forex"
    assert cfg.mode == "paper" and cfg.bucket_s == 900
    settings = SimpleNamespace(trading=SimpleNamespace(
        pipeline_symbols=["eurusd"], pipeline_market="forex",
        pipeline_mode="demo", pipeline_tick_spec="every 1h",
        pipeline_audit_spec="every 12h"))
    cfg = pipeline.pipeline_config(settings)
    assert cfg.symbols == ["EURUSD"] and cfg.mode == "demo"
    assert cfg.bucket_s == 3600 and cfg.audit_spec == "every 12h"


def test_control_state_folds_and_defaults_running(tmp_path):
    journal = tmp_path / "j.jsonl"
    assert pipeline.control_state(journal_path=journal)["state"] \
        == "running"
    out = pipeline.set_control("stop", actor="brain", reason="rest",
                               journal_path=journal, now=BASE)
    assert out["ok"] and out["state"] == "stopped"
    state = pipeline.control_state(journal_path=journal)
    assert state["state"] == "stopped" and state["actor"] == "brain"
    assert state["reason"] == "rest"
    out = pipeline.set_control("start", journal_path=journal,
                               now=BASE + 1)
    assert out["ok"] and pipeline.control_state(
        journal_path=journal)["state"] == "running"


# ── the tick: stages, idempotency, halts ─────────────────────────────

def test_tick_runs_every_stage_and_journals_once(tmp_path, stages):
    journal = tmp_path / "j.jsonl"
    engine = FakeEngine()
    out = pipeline.run_tick(journal_path=journal, now=BASE,
                            engine=engine, config=_cfg())
    assert out["state"] == "ran"
    stages_out = out["stages"]["XAUUSD"]
    assert stages_out["forecast"]["rungs"] == {
        tag: fc.METHOD for tag in fc.HORIZONS}
    assert stages_out["manage"]["state"] == "ran"
    assert engine.manage_calls == ["XAUUSD"]
    assert stages == {"read": 1, "shadow": 1, "judge": 1,
                      "forecast": 1, "resolve": 1}
    assert len(_ticks(journal)) == 1


def test_double_scheduled_tick_is_a_single_effect(tmp_path, stages):
    journal = tmp_path / "j.jsonl"
    engine = FakeEngine()
    first = pipeline.run_tick(journal_path=journal, now=BASE + 30,
                              engine=engine, config=_cfg())
    second = pipeline.run_tick(journal_path=journal, now=BASE + 60,
                               engine=engine, config=_cfg())
    assert first["state"] == "ran"
    assert second["state"] == "already_ran"
    assert second["tick_id"] == first["tick_id"]  # same 15m slot
    assert len(_ticks(journal)) == 1  # journaled once
    assert engine.manage_calls == ["XAUUSD"]  # acted once
    assert stages["shadow"] == 1
    # the NEXT slot runs again
    third = pipeline.run_tick(journal_path=journal, now=BASE + 901,
                              engine=engine, config=_cfg())
    assert third["state"] == "ran" and len(_ticks(journal)) == 2


def test_forced_tick_always_runs(tmp_path, stages):
    journal = tmp_path / "j.jsonl"
    engine = FakeEngine()
    pipeline.run_tick(journal_path=journal, now=BASE, engine=engine,
                      force=True, actor="brain", config=_cfg())
    out = pipeline.run_tick(journal_path=journal, now=BASE,
                            engine=engine, force=True, actor="brain",
                            config=_cfg())
    assert out["state"] == "ran" and len(_ticks(journal)) == 2


def test_stopped_pipeline_skips_every_stage(tmp_path, stages):
    journal = tmp_path / "j.jsonl"
    pipeline.set_control("stop", journal_path=journal, now=BASE)
    out = pipeline.run_tick(journal_path=journal, now=BASE + 1,
                            engine=FakeEngine(), config=_cfg())
    assert out["state"] == "stopped"
    assert stages == {"read": 0, "shadow": 0, "judge": 0,
                      "forecast": 0, "resolve": 0}
    assert len(_ticks(journal)) == 1  # the skip itself is journaled


def test_halt_blocks_tick_and_start_until_owner_reset(tmp_path,
                                                      stages):
    journal = tmp_path / "j.jsonl"
    halt = pipeline.halt_path(journal_path=journal)
    halt.write_text("halt", encoding="utf-8")
    out = pipeline.run_tick(journal_path=journal, now=BASE,
                            engine=FakeEngine(), config=_cfg())
    assert out["state"] == "halted"
    assert stages["read"] == 0  # nothing ran
    # start is refused while halted — and the refusal is journaled
    refused = pipeline.set_control("start", actor="brain",
                                   journal_path=journal, now=BASE)
    assert not refused["ok"] and "HALT" in refused["error"]
    controls = trading_journal.read_events(
        limit=0, path=journal, event="pipeline_control")
    assert any(c["result"] == "refused_halt" for c in controls)
    # stop is never refused, even halted
    assert pipeline.set_control("stop", journal_path=journal,
                                now=BASE)["ok"]
    # the owner's reset: remove the HALT file — then start applies
    halt.unlink()
    assert pipeline.set_control("start", journal_path=journal,
                                now=BASE)["ok"]


def test_manage_stage_reports_unavailable_rungs(tmp_path, stages):
    journal = tmp_path / "j.jsonl"
    # no engine at all (vault seam failed upstream)
    out = pipeline.run_tick(journal_path=journal, now=BASE,
                            engine=None, config=_cfg())
    assert out["state"] == "ran"  # the tick itself never crashes
    assert out["stages"]["XAUUSD"]["manage"]["state"] \
        == "rung_unavailable"
    # bridge rung: no desk, no management
    out = pipeline.run_tick(journal_path=journal, now=BASE + 1000,
                            engine=FakeEngine(desk=False,
                                              rung="mt5_bridge"),
                            config=_cfg())
    assert out["stages"]["XAUUSD"]["manage"]["state"] \
        == "rung_unavailable"
    # demo probe fails (absent credentials): honest state, and the
    # engine's manage is never even called
    engine = FakeEngine(rung="exness_demo")
    out = pipeline.run_tick(
        journal_path=journal, now=BASE + 2000, engine=engine,
        rung_probe=lambda: (False, "demo rung unavailable: "
                                   "ExnessError: not connected"),
        config=_cfg())
    manage = out["stages"]["XAUUSD"]["manage"]
    assert manage["state"] == "rung_unavailable"
    assert "not connected" in manage["detail"]
    assert engine.manage_calls == []


def test_manage_stage_error_is_reported_not_raised(tmp_path, stages):
    class BoomEngine(FakeEngine):
        def manage_pass(self, *, symbol=None):
            raise RuntimeError("desk exploded")

    out = pipeline.run_tick(journal_path=tmp_path / "j.jsonl",
                            now=BASE, engine=BoomEngine(),
                            config=_cfg())
    manage = out["stages"]["XAUUSD"]["manage"]
    assert manage["state"] == "error"
    assert "desk exploded" in manage["detail"]
    assert out["state"] == "ran"


# ── the audit cycle + status fold ────────────────────────────────────

def test_audit_cycle_journals_and_status_folds(tmp_path, stages,
                                               monkeypatch):
    journal = tmp_path / "j.jsonl"
    monkeypatch.setattr(ft, "train_all", lambda *a, **kw: [
        {"horizon": tag, "state": "insufficient", "fit_n": 0}
        for tag in fc.HORIZONS])
    out = pipeline.run_audit_cycle(journal_path=journal, now=BASE,
                                   config=_cfg())
    assert out["per_symbol"]["XAUUSD"]["verdicts"]["forecasts"] == {}
    audits = trading_journal.read_events(limit=0, path=journal,
                                         event="pipeline_audit")
    assert len(audits) == 1
    status = pipeline.pipeline_status(journal_path=journal, now=BASE)
    assert status["state"] == "running" and status["halted"] is False
    assert status["last_training"] is None  # train fake journals none
    rendered = pipeline.render_status(status)
    assert "trading pipeline — running" in rendered


# ── durable cadence: registration survives restarts ──────────────────

def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(home_path=tmp_path,
                                        workspace_dir=str(tmp_path)),
        extras={}, tools=None)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    return ctx, registry


def test_jobs_register_idempotently_and_survive_restart(tmp_path):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.finance.trading_jobs import ensure_trading_jobs

    ctx, _registry = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    out = ensure_trading_jobs(sched)
    assert [o["scheduled"] for o in out] == [True, True], out
    rows = {j["name"]: j for j in sched.list_jobs()}
    assert set(rows) == {"trading pipeline", "trading audit"}
    assert rows["trading pipeline"]["overlap_policy"] == "skip"
    assert rows["trading pipeline"]["routine"] is True
    import json as _json
    stored = {r["name"]: r for r in ctx.db.query(
        "SELECT name, spec, payload FROM schedule_jobs")}
    payload = _json.loads(stored["trading pipeline"]["payload"])
    assert payload == {"tool": "trading",
                       "args": {"action": "pipeline_tick"}}
    payload = _json.loads(stored["trading audit"]["payload"])
    assert payload == {"tool": "trading",
                       "args": {"action": "pipeline_audit"}}
    # re-registration is a no-op (boot after boot)
    out2 = ensure_trading_jobs(sched)
    assert all(o.get("already_scheduled") for o in out2), out2
    # a fresh scheduler over the same store sees the same jobs —
    # the cadence survived the "restart"
    sched2 = Scheduler(ctx)
    names = {j["name"] for j in sched2.list_jobs()}
    assert {"trading pipeline", "trading audit"} <= names
    out3 = ensure_trading_jobs(sched2)
    assert all(o.get("already_scheduled") for o in out3), out3
    assert len([j for j in sched2.list_jobs()
                if j["name"].startswith("trading")]) == 2


def _fake_frame(*, n=1500, base=2600.0, drift=0.35, tf="1h"):
    closes = [base + drift * i + 6.0 * math.sin(i * 2 * math.pi / 24)
              for i in range(n)]
    rows, idx = [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c - drift
        rows.append((o, max(o, c) + 0.9, min(o, c) - 0.9, c, 100.0))
        idx.append(pd.Timestamp(BASE - (n - 1 - i) * _STEP[tf],
                                unit="s", tz="UTC"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                     "volume"], index=idx)
    df.attrs["source"] = "fake"
    df.attrs["freshness"] = "fresh"
    df.attrs["last_bar_ts"] = BASE
    return df


def test_scheduled_tick_job_runs_the_real_pipeline(tmp_path,
                                                   monkeypatch):
    """The durable job fires the trading tool → the real organs run
    against faked market data and the tick lands in the journal."""
    from nomorals.agents.scheduler import Scheduler
    from nomorals.finance.trading_jobs import ensure_trading_jobs

    def ohlcv(symbol, market="crypto", timeframe="1h", bars=500,
              source="auto"):
        return _fake_frame(n=min(bars, 1500), tf=timeframe)

    def quote(symbol, market="crypto"):
        return {"symbol": symbol, "price": 2677.5,
                "change_pct_24h": 0.8, "currency": "USD",
                "source": "fake_quote", "quote_ts": BASE,
                "quote_age_s": 1.0, "freshness": "fresh",
                "bid": 2677.4, "ask": 2677.6}

    monkeypatch.setattr(market_data, "get_ohlcv", ohlcv)
    monkeypatch.setattr(market_data, "quote", quote)
    ctx, _registry = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = mock.MagicMock()
    ensure_trading_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == "trading pipeline"][0]
    result = sched.run_now(row["id"])
    assert result["ok"] is True, result
    journal = trading_journal.journal_path(ctx.settings)
    ticks = _ticks(journal)
    assert len(ticks) == 1 and ticks[0]["state"] == "ran"
    events = {r.get("event") for r in trading_journal.read_events(
        limit=0, path=journal)}
    assert {"market_read", "shadow_pass", "forecast",
            "judgment"} <= events


# ── the brain surface + CLI ──────────────────────────────────────────

class _FakeRegistry:
    def __init__(self, context):
        self.context = context
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


def _tool(tmp_path):
    from nomorals.tools import trading as trading_tool

    settings = SimpleNamespace(
        home_path=tmp_path,
        trading=SimpleNamespace(enabled=True))
    context = SimpleNamespace(settings=settings, extras={})
    reg = _FakeRegistry(context)
    trading_tool.register(reg)
    return reg.tools["trading"], settings


def test_brain_controls_the_pipeline(tmp_path, stages):
    trading, settings = _tool(tmp_path)
    journal = trading_journal.journal_path(settings)
    out = trading(action="pipeline_status")
    assert out["state"] == "running" and "trading pipeline" \
        in out["rendered"]
    out = trading(action="pipeline_stop", reason="owner rest")
    assert out["ok"] and out["state"] == "stopped"
    assert trading(action="pipeline_status")["state"] == "stopped"
    out = trading(action="pipeline_start")
    assert out["ok"] and out["state"] == "running"
    # a brain-poked tick runs the chain; the engine cannot build
    # over this bare context (no vault) → manage reports honestly
    out = trading(action="pipeline_tick")
    assert out["ok"] and out["state"] == "ran", out
    manage = out["stages"]["XAUUSD"]["manage"]
    assert manage["state"] == "rung_unavailable"
    ticks = _ticks(journal)
    assert len(ticks) == 1 and ticks[0]["actor"] == "brain"


def test_brain_start_refused_while_halted(tmp_path, stages):
    trading, settings = _tool(tmp_path)
    journal = trading_journal.journal_path(settings)
    halt = pipeline.halt_path(settings=settings)
    halt.parent.mkdir(parents=True, exist_ok=True)
    halt.write_text("halt", encoding="utf-8")
    out = trading(action="pipeline_start")
    assert out["ok"] is False and "HALT" in out["error"]
    # stop still works under the halt
    out = trading(action="pipeline_stop")
    assert out["ok"] and out["state"] == "stopped"
    assert journal.exists()


def test_cmdline_pipeline(tmp_path, capsys, stages):
    from nomorals.cmdline.commands.trade import _cmd_trade

    settings = SimpleNamespace(
        home_path=tmp_path,
        trading=SimpleNamespace(enabled=True))
    context = SimpleNamespace(settings=settings, extras={})
    args = argparse.Namespace(trade_action="pipeline",
                              pipeline_action="status", json=False)
    assert _cmd_trade(args, context) == 0
    assert "trading pipeline" in capsys.readouterr().out
    args = argparse.Namespace(trade_action="pipeline",
                              pipeline_action="stop", json=False)
    assert _cmd_trade(args, context) == 0
    state = pipeline.control_state(settings=settings)
    assert state["state"] == "stopped"


def test_parser_pipeline_actions():
    from nomorals.cmdline.parser import _parser

    for action in ("status", "start", "stop", "tick", "audit"):
        args = _parser().parse_args(["trade", "pipeline", action])
        assert args.trade_action == "pipeline"
        assert args.pipeline_action == action
