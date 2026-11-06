"""Trading pipeline — Devon trading system, Pass 6 (brain control
+ autonomy).

Passes 1–5 built organs that only move when poked (``nm trade`` /
the ``trading`` tool). This module is the standing loop that runs
them, on the seams Devon already has — no parallel scheduler:

* **Cadence** rides the durable DB-backed Scheduler
  (:mod:`nomorals.finance.trading_jobs` registers two jobs whose
  payloads call the ``trading`` tool's ``pipeline_tick`` /
  ``pipeline_audit`` actions; jobs persist in the schedule store,
  so the cadence survives restarts and re-registration is
  idempotent by name).
* **State** folds from the Pass 1 trading journal, like every
  other organ: ``pipeline_control`` events carry start/stop,
  ``pipeline_tick`` events carry each tick's stages, the Pass 5
  ``trade_audit`` events carry the verdicts. No new store, no
  migration.
* **Brain control** is the ``trading`` tool's pipeline actions
  (tools/trading.py) calling :func:`set_control` / :func:`run_tick`
  / :func:`pipeline_status` here — natural-language reachable
  through the same tool surface as every other system.

One tick, per configured symbol, in order: a fresh whole-market
read → the shadow pass (setup detection + lifecycle) → due
forecast outcomes resolved → a forecast pass (trained rung where a
fresh fitted model exists, empirical otherwise) → the shadow
judgment pass → the execution-desk MANAGEMENT poll (open positions
managed under their Pass 4 profiles). **The pipeline places no new
orders**: execution of fresh setups stays the Pass 4 owner-poked
``execute`` action — judgment remains shadow-first and this pass
grants no order authority anywhere.

Idempotency: a scheduled tick is keyed by its cadence slot
(``floor(now / bucket)`` per symbol set). A doubled fire — retry,
catch-up, overlap — folds the journal, sees the slot's tick
record, and returns ``already_ran`` without re-running a stage or
journaling a second record. Brain-poked ticks (``force=True``)
are keyed by their exact start time and always run.

Halts outrank everything: while the desk HALT file (the Pass 4
kill switch, shared finance home) is present, a tick journals a
``halted`` record and runs nothing, and a pipeline START is
refused — the system stays halted until the owner's own reset
path (remove the HALT file; the desk states it on its face). Stop
is never refused: halting the pipeline is always allowed.

Demo seam: the management poll's engine is built by the caller
(the tools layer's ``_system_engine`` over the vault). Paper mode
is the simulator rung and always available; demo mode probes the
broker snapshot first — absent vault credentials surface as the
stage's honest ``rung_unavailable`` with the connector's own
message, the rest of the tick still runs, and nothing anywhere
fabricates a fill.

Placement: ``agents/`` beside the organs it drives.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..finance import trading_journal
from . import forecast as forecast_mod
from . import forecast_trained, market_read
from . import judgment as judgment_mod
from . import setups as setups_mod
from . import trade_audit as audit_mod

_log = logging.getLogger(__name__)

#: Defaults (settings.trading.pipeline_* override each).
DEFAULT_SYMBOLS: tuple[str, ...] = ("XAUUSD",)
DEFAULT_MARKET = "forex"
DEFAULT_MODE = "paper"
DEFAULT_TICK_SPEC = "every 15m"
DEFAULT_AUDIT_SPEC = "every 6h"
#: Fallback slot bucket when the tick spec can't be parsed.
DEFAULT_BUCKET_S = 900


@dataclass
class PipelineConfig:
    symbols: list[str] = field(
        default_factory=lambda: list(DEFAULT_SYMBOLS))
    market: str = DEFAULT_MARKET
    mode: str = DEFAULT_MODE
    tick_spec: str = DEFAULT_TICK_SPEC
    audit_spec: str = DEFAULT_AUDIT_SPEC
    bucket_s: int = DEFAULT_BUCKET_S


def _bucket_from_spec(spec: str) -> int:
    match = re.fullmatch(r"every\s+(\d+)\s*([mh])", (spec or "").strip())
    if not match:
        return DEFAULT_BUCKET_S
    n = int(match.group(1))
    return n * 60 if match.group(2) == "m" else n * 3600


def pipeline_config(settings: Any = None) -> PipelineConfig:
    """The pipeline config from ``settings.trading`` (getattr
    chain — every field optional, defaults stated above)."""
    trading = getattr(settings, "trading", None)

    def get(name: str, default: Any) -> Any:
        value = getattr(trading, name, None)
        return default if value is None else value

    symbols = [str(s).strip().upper()
               for s in get("pipeline_symbols", DEFAULT_SYMBOLS)
               if str(s).strip()] or list(DEFAULT_SYMBOLS)
    tick_spec = str(get("pipeline_tick_spec", DEFAULT_TICK_SPEC))
    return PipelineConfig(
        symbols=symbols,
        market=str(get("pipeline_market", DEFAULT_MARKET)),
        mode=str(get("pipeline_mode", DEFAULT_MODE)).lower(),
        tick_spec=tick_spec,
        audit_spec=str(get("pipeline_audit_spec", DEFAULT_AUDIT_SPEC)),
        bucket_s=_bucket_from_spec(tick_spec))


# ── the halt seam (Pass 4's kill switch, respected, never reset) ─────

def halt_path(settings: Any = None,
              journal_path: Any = None) -> Path:
    """The desk HALT file: the finance home beside the journal
    (the system desk's state lives in the same home — one switch,
    as Pass 4 built it)."""
    if journal_path is not None:
        return Path(journal_path).parent / "HALT"
    return trading_journal.journal_path(settings=settings).parent / "HALT"


def halt_engaged(settings: Any = None,
                  journal_path: Any = None) -> bool:
    return halt_path(settings=settings,
                     journal_path=journal_path).exists()


# ── control state: journaled start/stop, folded ──────────────────────

def control_state(settings: Any = None, journal_path: Any = None
                  ) -> dict[str, Any]:
    """Latest pipeline_control event wins; no event = running
    (the standing posture: the system runs unattended until the
    owner stands it down)."""
    for rec in trading_journal.read_events(
            limit=0, event="pipeline_control", settings=settings,
            path=journal_path):
        return {"state": rec.get("state") or "running",
                "since": rec.get("ts"), "actor": rec.get("actor"),
                "reason": rec.get("reason") or "",
                "result": rec.get("result") or "applied"}
    return {"state": "running", "since": None, "actor": None,
            "reason": "", "result": "default"}


def set_control(action: str, *, actor: str = "brain",
                reason: str = "", settings: Any = None,
                journal_path: Any = None, now: float | None = None
                ) -> dict[str, Any]:
    """Start/stop the pipeline. A journaled control event either
    way — including a refused start, so the journal shows the
    attempt and its reason.

    Start is refused while the HALT kill switch is engaged (the
    system stays halted until the owner's reset: remove the HALT
    file). Stop is always applied — safety never refuses.
    """
    taken = float(now if now is not None else time.time())
    action = (action or "").strip().lower()
    if action not in ("start", "stop"):
        return {"ok": False, "error": f"unknown control action "
                                      f"{action!r} (start|stop)"}
    if action == "start" and halt_engaged(
            settings=settings, journal_path=journal_path):
        trading_journal.append_event(
            "pipeline_control", settings=settings, path=journal_path,
            state="stopped", action="start", result="refused_halt",
            actor=actor, reason=reason, ts=taken)
        return {"ok": False, "state": "stopped",
                "error": "kill switch engaged (HALT file present) — "
                         "the pipeline stays halted until the HALT "
                         "file is removed (the owner's reset path)"}
    state = "running" if action == "start" else "stopped"
    trading_journal.append_event(
        "pipeline_control", settings=settings, path=journal_path,
        state=state, action=action, result="applied", actor=actor,
        reason=reason, ts=taken)
    return {"ok": True, "state": state, "actor": actor,
            "reason": reason, "ts": taken}


# ── one tick ─────────────────────────────────────────────────────────

def _slot_for(taken: float, bucket_s: int) -> int:
    return int(taken // bucket_s) * bucket_s


def _tick_seen(tick_id: str, *, settings: Any, journal_path: Any
               ) -> dict[str, Any] | None:
    for rec in trading_journal.read_events(
            limit=0, event="pipeline_tick", settings=settings,
            path=journal_path):
        if rec.get("tick_id") == tick_id:
            return rec
    return None


def _forecast_stage(symbol: str, cfg: PipelineConfig, *,
                    settings: Any, journal_path: Any, taken: float
                    ) -> dict[str, Any]:
    """Resolve due outcomes, then one forecast pass — the trained
    rung where a fresh fitted model exists for any horizon, else
    the empirical pass (which also merges any GS seed samples the
    owner configured)."""
    resolved = forecast_mod.resolve_due_forecasts(
        symbol, market=cfg.market, settings=settings,
        journal_path=journal_path, now=taken)
    has_model = any(
        forecast_trained.latest_model(
            symbol, tag, settings=settings,
            journal_path=journal_path, now=taken) is not None
        for tag in forecast_mod.HORIZONS)
    if has_model:
        records = forecast_trained.make_trained_forecasts(
            symbol, market=cfg.market, settings=settings,
            journal_path=journal_path, now=taken)
    else:
        seed = None
        seed_path = getattr(getattr(settings, "trading", None),
                            "forecast_seed_path", None)
        if seed_path:
            seed = forecast_mod.load_seed_samples(seed_path)
        records = forecast_mod.make_forecasts(
            symbol, market=cfg.market, settings=settings,
            journal_path=journal_path, now=taken, seed_samples=seed)
    return {"state": "ran", "resolved_outcomes": resolved,
            "rungs": {r["horizon"]: r.get("method")
                      for r in records},
            "directions": {r["horizon"]: r.get("direction")
                           for r in records}}


def _manage_stage(engine: Any, rung_probe: Callable[[], tuple[bool, str]] | None,
                  symbol: str) -> dict[str, Any]:
    """The execution-desk management poll for one symbol."""
    if engine is None:
        return {"state": "rung_unavailable",
                "detail": "execution engine could not be built "
                          "(vault/connector seam — see log)"}
    if getattr(engine, "desk", None) is None:
        return {"state": "rung_unavailable", "rung": engine.rung,
                "detail": "bridge rung — Pass 4 manages no positions "
                          "on it (execution reports unavailable there "
                          "too)"}
    if rung_probe is not None:
        ok, detail = rung_probe()
        if not ok:
            return {"state": "rung_unavailable", "rung": engine.rung,
                    "detail": detail}
    try:
        rep = engine.manage_pass(symbol=symbol)
    except Exception as exc:  # noqa: BLE001 - a stage reports, never crashes the tick
        return {"state": "error", "rung": engine.rung,
                "detail": f"{type(exc).__name__}: {exc}"}
    return {"state": "ran", "rung": engine.rung,
            "counts": rep.get("counts")}


def run_tick(*, settings: Any = None, journal_path: Any = None,
             now: float | None = None, engine: Any = None,
             rung_probe: Callable[[], tuple[bool, str]] | None = None,
             force: bool = False, actor: str = "scheduler",
             config: PipelineConfig | None = None
             ) -> dict[str, Any]:
    """One pipeline tick over every configured symbol.

    Scheduled ticks (``force=False``) are slot-keyed and
    idempotent: a slot whose tick record already exists returns
    ``already_ran`` with no stage re-run and no second record.
    """
    taken = float(now if now is not None else time.time())
    cfg = config or pipeline_config(settings)
    slot = int(taken) if force else _slot_for(taken, cfg.bucket_s)
    tick_id = f"pipe:{slot}:{'/'.join(cfg.symbols)}"
    if not force:
        seen = _tick_seen(tick_id, settings=settings,
                          journal_path=journal_path)
        if seen is not None:
            return {"kind": "pipeline_tick", "tick_id": tick_id,
                    "slot": slot, "state": "already_ran",
                    "original_state": seen.get("state"),
                    "taken_at": taken}
    base: dict[str, Any] = {"kind": "pipeline_tick",
                            "tick_id": tick_id, "slot": slot,
                            "taken_at": taken, "actor": actor,
                            "symbols": cfg.symbols,
                            "market": cfg.market, "mode": cfg.mode}
    control = control_state(settings=settings,
                            journal_path=journal_path)
    if control["state"] != "running":
        result = {**base, "state": "stopped",
                  "detail": "pipeline is stopped (control state) — "
                            "tick skipped"}
        trading_journal.append_event(
            "pipeline_tick", settings=settings, path=journal_path,
            **result)
        return result
    if halt_engaged(settings=settings, journal_path=journal_path):
        result = {**base, "state": "halted",
                  "detail": "kill switch engaged (HALT file present)"
                            " — no stage ran; the system stays "
                            "halted until the HALT file is removed"}
        trading_journal.append_event(
            "pipeline_tick", settings=settings, path=journal_path,
            **result)
        return result
    per_symbol: dict[str, Any] = {}
    for symbol in cfg.symbols:
        stages: dict[str, Any] = {}
        read = market_read.whole_market_read(
            symbol, market=cfg.market, journal=True,
            settings=settings, journal_path=journal_path, now=taken)
        shadow = setups_mod.shadow_pass(
            symbol, market=cfg.market, settings=settings,
            journal_path=journal_path, now=taken, read=read)
        stages["shadow"] = {
            "state": "ran",
            "counts": {k: len(shadow[k]) for k in
                       ("created", "armed", "triggered",
                        "invalidated", "expired")},
            "detector_errors": shadow.get("detector_errors") or []}
        stages["forecast"] = _forecast_stage(
            symbol, cfg, settings=settings,
            journal_path=journal_path, taken=taken)
        judged = judgment_mod.judgment_pass(
            symbol, market=cfg.market, settings=settings,
            journal_path=journal_path, now=taken, read=read)
        stages["judgment"] = {"state": "ran",
                              "counts": judged.get("counts")}
        stages["manage"] = _manage_stage(engine, rung_probe, symbol)
        per_symbol[symbol] = stages
    result = {**base, "state": "ran", "stages": per_symbol}
    trading_journal.append_event(
        "pipeline_tick", settings=settings, path=journal_path,
        **result)
    return result


# ── the audit cycle ──────────────────────────────────────────────────

def run_audit_cycle(*, settings: Any = None, journal_path: Any = None,
                    now: float | None = None, retrain: bool = True,
                    config: PipelineConfig | None = None
                    ) -> dict[str, Any]:
    """One audit cycle per configured symbol: refit the trained
    estimators, grade the journal (Pass 5 — journals its own
    ``trade_audit`` event), and compare the forecast rungs in the
    audit's terms. The cycle itself journals one ``pipeline_audit``
    summary. Verdicts remain recommendations; nothing here
    promotes, gates or executes."""
    taken = float(now if now is not None else time.time())
    cfg = config or pipeline_config(settings)
    per_symbol: dict[str, Any] = {}
    for symbol in cfg.symbols:
        entry: dict[str, Any] = {}
        if retrain:
            trained = forecast_trained.train_all(
                symbol, market=cfg.market, settings=settings,
                journal_path=journal_path, now=taken)
            entry["training"] = {
                r["horizon"]: {"state": r["state"],
                               "fit_n": r["fit_n"]}
                for r in trained}
        report = audit_mod.run_audit(
            symbol, settings=settings, journal_path=journal_path,
            now=taken)
        entry["verdicts"] = report["verdicts"]
        entry["comparison"] = forecast_trained.rung_comparison(
            symbol, settings=settings, journal_path=journal_path)
        per_symbol[symbol] = entry
    result: dict[str, Any] = {
        "kind": "pipeline_audit", "taken_at": taken,
        "symbols": cfg.symbols, "per_symbol": per_symbol}
    trading_journal.append_event(
        "pipeline_audit", settings=settings, path=journal_path,
        taken_at=taken,
        verdicts={sym: e["verdicts"] for sym, e in per_symbol.items()},
        training={sym: e.get("training") for sym, e in
                  per_symbol.items()})
    return result


# ── status: the brain's inspect surface ──────────────────────────────

def pipeline_status(*, settings: Any = None, journal_path: Any = None,
                    scheduler: Any = None, now: float | None = None
                    ) -> dict[str, Any]:
    """Folded pipeline state: control, halt, cadence jobs (when a
    scheduler is attached), last tick, last audit verdicts, last
    training run."""
    taken = float(now if now is not None else time.time())
    cfg = pipeline_config(settings)
    control = control_state(settings=settings,
                            journal_path=journal_path)
    jobs = None
    if scheduler is not None:
        jobs = [{"name": j.get("name"), "spec": j.get("spec"),
                 "next_run": j.get("next_run"),
                 "enabled": j.get("enabled", True)}
                for j in scheduler.list_jobs()
                if j.get("name") in ("trading pipeline",
                                     "trading audit")]
    last_tick = None
    for rec in trading_journal.read_events(
            limit=0, event="pipeline_tick", settings=settings,
            path=journal_path):
        last_tick = {"tick_id": rec.get("tick_id"),
                     "state": rec.get("state"),
                     "taken_at": rec.get("ts"),
                     "actor": rec.get("actor")}
        break
    last_audit = None
    for rec in trading_journal.read_events(
            limit=0, event="trade_audit", settings=settings,
            path=journal_path):
        last_audit = {"taken_at": rec.get("taken_at"),
                      "symbol": rec.get("symbol"),
                      "verdicts": rec.get("verdicts")}
        break
    last_training = None
    for rec in trading_journal.read_events(
            limit=0, event="forecast_training", settings=settings,
            path=journal_path):
        last_training = {"trained_at": rec.get("trained_at"),
                         "symbol": rec.get("symbol"),
                         "horizons": rec.get("horizons")}
        break
    return {"kind": "pipeline_status", "taken_at": taken,
            "state": control["state"], "control": control,
            "halted": halt_engaged(settings=settings,
                                   journal_path=journal_path),
            "config": {"symbols": cfg.symbols, "market": cfg.market,
                       "mode": cfg.mode, "tick_spec": cfg.tick_spec,
                       "audit_spec": cfg.audit_spec},
            "jobs": jobs, "last_tick": last_tick,
            "last_audit": last_audit, "last_training": last_training}


# ── surfaces ─────────────────────────────────────────────────────────

def _iso(taken: float | None) -> str:
    if taken is None:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(float(taken)))


def render_status(status: dict[str, Any]) -> str:
    cfg = status["config"]
    lines = [
        f"trading pipeline — {status['state']}"
        + (" · HALTED (kill switch)" if status["halted"] else ""),
        f"symbols {', '.join(cfg['symbols'])} · market {cfg['market']}"
        f" · desk rung {cfg['mode']} · tick {cfg['tick_spec']}"
        f" · audit {cfg['audit_spec']}"]
    if status["jobs"] is None:
        lines.append("cadence jobs: scheduler not attached in this "
                     "process (jobs persist in the schedule store)")
    elif not status["jobs"]:
        lines.append("cadence jobs: NOT registered — the runtime "
                     "registers them at boot")
    else:
        for job in status["jobs"]:
            lines.append(f"cadence job: {job['name']} "
                         f"({job.get('spec')})")
    tick = status["last_tick"]
    lines.append(f"last tick: {_iso(tick['taken_at']) if tick else 'never'}"
                 + (f" — {tick['state']} (by {tick['actor']})"
                    if tick else ""))
    audit = status["last_audit"]
    if audit:
        flat = [f"{k}={v}" for group in (audit["verdicts"] or {}).values()
                for k, v in group.items()]
        lines.append(f"last audit: {_iso(audit['taken_at'])} — "
                     + (", ".join(flat) if flat else "no verdicts yet")
                     + " (recommendations only)")
    else:
        lines.append("last audit: never")
    training = status["last_training"]
    if training:
        states = ", ".join(
            f"{tag} {h.get('state')}" for tag, h in
            (training["horizons"] or {}).items())
        lines.append(f"last training: {_iso(training['trained_at'])}"
                     f" — {states}")
    return "\n".join(lines)


def render_tick(result: dict[str, Any]) -> str:
    head = (f"pipeline tick {result['tick_id']} — {result['state']}")
    if result["state"] != "ran":
        return head + (f"\n{result['detail']}"
                       if result.get("detail") else "")
    lines = [head]
    for symbol, stages in result["stages"].items():
        fc_stage = stages["forecast"]
        lines.append(
            f"{symbol}: setups "
            + " ".join(f"{k} {v}" for k, v in
                       stages["shadow"]["counts"].items())
            + " · forecasts "
            + ", ".join(f"{tag} {d or '—'}"
                        for tag, d in fc_stage["directions"].items())
            + f" ({fc_stage['resolved_outcomes']} outcomes resolved)"
            + f" · judged {stages['judgment']['counts']}"
            + f" · manage {stages['manage']['state']}")
    return "\n".join(lines)


def render_audit_cycle(result: dict[str, Any]) -> str:
    lines = [f"pipeline audit cycle — {_iso(result['taken_at'])} UTC"]
    for symbol, entry in result["per_symbol"].items():
        flat = [f"{k}={v}" for group in entry["verdicts"].values()
                for k, v in group.items()]
        lines.append(f"{symbol}: verdicts "
                     + (", ".join(flat) if flat else "none yet")
                     + " (recommendations only)")
        if entry.get("training"):
            lines.append("  training: " + ", ".join(
                f"{tag} {t['state']} (fit n={t['fit_n']})"
                for tag, t in entry["training"].items()))
        lines.append(forecast_trained.render_comparison(
            entry["comparison"]))
    return "\n".join(lines)
