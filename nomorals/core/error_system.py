"""Error system wiring — one bundle, one boot hook, zero-cost when unwired.

The machinery lives in :mod:`incidents`, :mod:`selfheal`, :mod:`budgets`
and :mod:`degradation`. This module is the *wiring*: it builds the bundle
with a persistent journal, exposes a module-level accessor so subsystems
(LLM router, telegram, voice, scheduler) can record heartbeats without
importing each other, and declares the default degradation ladders and
supervised workers.

Subsystems call :func:`heartbeat` / :func:`record_error`. Both are no-ops
until :func:`set_error_system` runs at boot — so importing this module
costs nothing and un-wired code paths stay fast.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .budgets import BudgetManager
from .degradation import DegradationLadder, LadderManager, Rung
from .incidents import IncidentJournal
from .logging_setup import get_logger
from .selfheal import SelfHealingExecutor, SubsystemSupervisor
from ..storage.kv import KVStore

_log = get_logger(__name__)


@dataclass
class ErrorSystem:
    """The whole error-catching apparatus, one object."""

    journal: IncidentJournal
    budgets: BudgetManager
    ladders: LadderManager
    supervisors: dict[str, SubsystemSupervisor] = field(default_factory=dict)
    _healers: dict[str, SelfHealingExecutor] = field(default_factory=dict)

    def supervisor_for(self, subsystem: str) -> SubsystemSupervisor:
        with _lock:
            sup = self.supervisors.get(subsystem)
            if sup is None:
                sup = SubsystemSupervisor(subsystem, journal=self.journal)
                self.supervisors[subsystem] = sup
            return sup

    def healer_for(self, subsystem: str) -> SelfHealingExecutor:
        """Per-subsystem self-healing executor (created on demand)."""
        with _lock:
            h = self._healers.get(subsystem)
            if h is None:
                h = SelfHealingExecutor(subsystem, journal=self.journal)
                self._healers[subsystem] = h
            return h

    def health(self) -> dict[str, Any]:
        """One snapshot for dashboards and the system_health tool."""
        try:
            alerts = self.budgets.check_all()
        except Exception:  # noqa: BLE001 - health must never raise
            alerts = []
        try:
            budget_status = self.budgets.status_all()
        except Exception:  # noqa: BLE001
            budget_status = {}
        try:
            ladder_status = self.ladders.status_all()
            degraded = self.ladders.degraded_subsystems()
        except Exception:  # noqa: BLE001
            ladder_status, degraded = {}, []
        try:
            top_failing = self.journal.top_failing(window_s=86400)
        except Exception:  # noqa: BLE001
            top_failing = []
        ok = not alerts and not degraded
        return {
            "ok": ok,
            "alerts": [
                {"subsystem": a.subsystem, "rule": a.rule_name,
                 "severity": a.severity, "detail": a.detail}
                for a in alerts
            ],
            "degraded_subsystems": degraded,
            "budgets": budget_status,
            "ladders": ladder_status,
            "top_failing_24h": [
                {"subsystem": s, "failures": n} for s, n in top_failing[:10]
            ],
            "supervised": sorted(self.supervisors),
        }

    def close(self) -> None:
        for sup in list(self.supervisors.values()):
            try:
                sup.stop_all()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.journal.close()
        except Exception:  # noqa: BLE001
            pass


def build_error_system(settings: Any = None) -> ErrorSystem:
    """Build the bundle with a persistent journal.

    Journal lives at ``<home>/data/error_journal.db`` (next to the main
    DB) so incident memory survives restarts. ``settings`` may be None in
    tests — then the journal falls back to ``:memory:``.
    """
    journal_path = ":memory:"
    if settings is not None:
        try:
            home = Path(os.path.expanduser(
                getattr(settings, "home", "~/.nomorals"))).resolve()
            data_dir = home / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            journal_path = str(data_dir / "error_journal.db")
        except Exception as exc:  # noqa: BLE001 - never break boot
            _log.warning("error journal dir unavailable (%s); using :memory:", exc)
            journal_path = ":memory:"
    journal = IncidentJournal(path=journal_path)
    budgets = BudgetManager(journal=journal, on_alert=_on_budget_alert)
    ladders = LadderManager()
    es = ErrorSystem(journal=journal, budgets=budgets, ladders=ladders)
    register_default_ladders(es)
    _log.info("error system built (journal=%s)", journal_path)
    return es


def _on_budget_alert(alert: Any) -> None:
    _log.warning("error budget alert: %s/%s %s",
                 getattr(alert, "subsystem", "?"),
                 getattr(alert, "rule_name", "?"),
                 getattr(alert, "detail", ""))


# ── module-level current system ──────────────────────────────────────────

_lock = threading.RLock()
_current: ErrorSystem | None = None


def set_error_system(es: ErrorSystem | None) -> None:
    global _current
    with _lock:
        _current = es


def get_error_system() -> ErrorSystem | None:
    with _lock:
        return _current


def heartbeat(subsystem: str, ok: bool) -> None:
    """Record one success/failure event. No-op until boot wires the system."""
    es = get_error_system()
    if es is None:
        return
    try:
        es.journal.record_heartbeat(subsystem, ok=ok)
    except Exception:  # noqa: BLE001 - telemetry must never break callers
        pass


def record_error(subsystem: str, exc: BaseException,
                 context: dict[str, Any] | None = None) -> None:
    """Record an incident + feed the healer. No-op until boot wires it."""
    es = get_error_system()
    if es is None:
        return
    try:
        es.journal.record_incident(exc, subsystem=subsystem,
                                   context=context or {})
    except Exception:  # noqa: BLE001
        pass


# ── default degradation ladders ──────────────────────────────────────────

def _llm_probe(router: Any) -> Callable[[], bool]:
    def probe() -> bool:
        try:
            return bool(router._chain())
        except Exception:  # noqa: BLE001
            return False
    return probe


def register_default_ladders(es: ErrorSystem, context: Any = None) -> None:
    """Declare ladders for the real subsystems.

    Rung 0 is always the primary path; lower rungs degrade honestly.
    Handlers receive the same args the primary got.
    """
    journal = es.journal

    # ── LLM: provider chain as a ladder ──
    def _llm_primary(prompt: str, **kw: Any) -> Any:
        router = getattr(context, "router", None) if context else None
        if router is None:
            raise RuntimeError("no LLM router on context")
        from ..llm.base import Message
        msgs = kw.get("messages") or [Message.user(prompt)]
        return router.chat(msgs, **{k: v for k, v in kw.items()
                                    if k != "messages"})

    def _llm_cached(prompt: str, **kw: Any) -> Any:
        raise RuntimeError(
            "all LLM providers unavailable and no cached answer for this prompt")

    es.ladders.register(DegradationLadder("llm", rungs=[
        Rung("primary", _llm_primary,
             probe=(_llm_probe(getattr(context, "router", None))
                    if context and getattr(context, "router", None) else None)),
        Rung("cached", _llm_cached,
             capability_trade="no live model; answers unavailable",
             honesty="All language models are unreachable right now — "
                     "I can't generate a fresh answer."),
    ], journal=journal))

    # ── Telegram: send → queue-for-retry ──
    def _tg_primary(chat: str, text: str, **kw: Any) -> Any:
        gw = getattr(context, "gateway", None) if context else None
        if gw is None:
            raise RuntimeError("no chat gateway on context")
        return gw.send("telegram", chat, text, **kw)

    def _tg_queued(chat: str, text: str, **kw: Any) -> Any:
        # Last rung: persist the message so a later tick can deliver it.
        db = getattr(context, "db", None) if context else None
        if db is None:
            raise RuntimeError("no database to queue the message")
        try:
            KVStore(db).set(f"tg_retry:{time.time()}:{chat}",
                            {"chat": chat, "text": text})
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not queue telegram message: {exc}")
        return {"ok": False, "queued": True, "chat": chat}

    es.ladders.register(DegradationLadder("telegram", rungs=[
        Rung("send", _tg_primary),
        Rung("queued", _tg_queued,
             capability_trade="message queued, not delivered",
             honesty="Telegram is unreachable — your message is queued and "
                     "will be delivered when the connection recovers."),
    ], journal=journal))

    # ── Voice: best backend → any backend → honest failure ──
    def _voice_primary(text: str, **kw: Any) -> Any:
        raise RuntimeError("voice primary rung needs the TTS engine; "
                           "use the voice tools directly")

    def _voice_silent(text: str, **kw: Any) -> Any:
        return {"ok": False, "silent": True,
                "note": "voice synthesis unavailable"}

    es.ladders.register(DegradationLadder("voice", rungs=[
        Rung("tts", _voice_primary),
        Rung("silent", _voice_silent,
             capability_trade="no audio produced",
             honesty="Voice synthesis is unavailable right now — "
                     "here's the text instead."),
    ], journal=journal))

    _log.info("default degradation ladders registered: llm, telegram, voice")
