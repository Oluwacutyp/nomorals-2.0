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
from .failures import record_failure
from .incidents import IncidentJournal
from .logging_setup import get_logger
from .selfheal import SelfHealingExecutor, SubsystemSupervisor

_log = get_logger(__name__)


@dataclass
class ErrorSystem:
    """The whole error-catching apparatus, one object."""

    journal: IncidentJournal
    budgets: BudgetManager
    ladders: LadderManager
    #: The agent context the ladders were wired against (composition
    #: root). Callers that wrap the context's own resources (e.g. the
    #: llm Brain over ``context.router``) check identity against this
    #: before routing through a ladder — a ladder's rung 0 IS the
    #: context's router/gateway, never a private chain.
    context: Any = None
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
        try:
            top_subsystems = self.journal.top_subsystems(window_s=86400)
        except Exception:  # noqa: BLE001
            top_subsystems = []
        try:
            mttr = self.journal.mttr(window_s=86400)
        except Exception:  # noqa: BLE001
            mttr = {}
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
                {"signature": f.get("signature"), "subsystem": f.get("subsystem"),
                 "failures": f.get("n"), "last": f.get("last_ts")}
                for f in top_failing[:10]
            ],
            "top_subsystems_24h": top_subsystems[:10],
            "mttr_24h": mttr,
            "supervised": sorted(self.supervisors),
        }

    def format_health(self, *, color: bool | None = None,
                      theme: Any = None) -> str:
        """A styled one-screen health report for chat and dashboards."""
        from .style import paint, supports_color, status_dot, header

        if color is None:
            color = supports_color()
        snap = self.health()
        dot = status_dot("ok" if snap["ok"] else "error", theme, color=color)
        lines = [header(f"{dot} system health", theme, color=color), ""]
        alerts = snap["alerts"]
        if alerts:
            lines.append(paint("ALERTS", "label", theme, color=color))
            for a in alerts[:8]:
                lines.append(f"  {paint(a['subsystem'], 'error', theme, color=color)}"
                             f" [{a['severity']}] {a['rule']}: {a['detail']}")
            lines.append("")
        degraded = snap["degraded_subsystems"]
        if degraded:
            lines.append(paint("DEGRADED", "label", theme, color=color))
            for d in degraded:
                lines.append(f"  {paint(str(d), 'warn', theme, color=color)}")
            lines.append("")
        top = snap["top_subsystems_24h"]
        if top:
            lines.append(paint("HOT SUBSYSTEMS (24h)", "label", theme, color=color))
            for s in top[:6]:
                lines.append(
                    f"  {paint(s['subsystem'], 'info', theme, color=color)}: "
                    f"{s['n']} incidents ({s['share']:.0%} of total), "
                    f"worst {s['worst_severity']}")
            lines.append("")
        mttr = snap["mttr_24h"] or {}
        if mttr.get("recovered"):
            lines.append(paint("RECOVERY (24h)", "label", theme, color=color))
            lines.append(f"  MTTR {mttr['mean_human']} (median {mttr['median_human']}), "
                         f"{mttr['recovered']} recovered / {mttr['open']} open")
            lines.append("")
        if snap["ok"] and not top:
            lines.append(paint("all quiet — no alerts, no incidents in 24h",
                               "ok", theme, color=color))
        return "\n".join(lines).rstrip()

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


def build_error_system(
    settings: Any = None,
    *,
    context: Any = None,
    kv_factory: Callable[[Any], Any] | None = None,
    message_factory: Callable[[str], Any] | None = None,
    tts_factory: Callable[[Any, str], Any] | None = None,
    tts_probe: Callable[[], bool] | None = None,
) -> ErrorSystem:
    """Build the bundle with a persistent journal.

    Journal lives at ``<home>/data/error_journal.db`` (next to the main
    DB) so incident memory survives restarts. ``settings`` may be None in
    tests — then the journal falls back to ``:memory:``.

    ``context`` is the agent context under construction: the ladder
    handlers resolve ``router`` / ``gateway`` / ``db`` from it LAZILY at
    call time, so resources attached after this build (the router is
    built before, the gateway only exists once the partner runtime
    starts) are still reachable. Without it the ladders register but
    their rung 0s can only fail honestly ("no LLM router on context").

    ``kv_factory`` / ``message_factory`` / ``tts_factory`` /
    ``tts_probe`` are injected by the composition root (see
    :func:`register_default_ladders`): the kernel may not import the
    storage / llm / voice layers. None means the rungs that need them
    fail honestly instead of persisting / building messages.
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
    es = ErrorSystem(journal=journal, budgets=budgets, ladders=ladders,
                     context=context)
    register_default_ladders(es, context, kv_factory=kv_factory,
                             message_factory=message_factory,
                             tts_factory=tts_factory, tts_probe=tts_probe)
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
    except Exception as exc:  # noqa: BLE001 - telemetry must never break callers
        record_failure("error_system.heartbeat", exc)


def record_error(subsystem: str, exc: BaseException,
                 context: dict[str, Any] | None = None) -> None:
    """Record an incident + feed the healer. No-op until boot wires it."""
    es = get_error_system()
    if es is None:
        return
    try:
        es.journal.record_incident(exc, subsystem=subsystem,
                                   context=context or {})
    except Exception as exc:  # noqa: BLE001
        record_failure("error_system.record_error", exc)


# ── default degradation ladders ──────────────────────────────────────────

def _llm_probe(router: Any) -> Callable[[], bool]:
    def probe() -> bool:
        try:
            return bool(router._chain())
        except Exception:  # noqa: BLE001
            return False
    return probe


def _resolve_gateway(context: Any) -> Any | None:
    """Find the live chat gateway for a context (duck-typed mirror of
    ``agents.notifier.resolve_gateway`` — the kernel may not import the
    agents layer). The partner runtime stores the gateway in
    ``context.extras["gateway"]``; older/CLI code may set
    ``context.gateway`` directly. None when no gateway exists yet (it
    attaches after the error system is built — resolution is lazy, at
    call time, for exactly that reason)."""
    if context is None:
        return None
    gw = getattr(context, "gateway", None)
    if gw is not None:
        return gw
    try:
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            return extras.get("gateway")
    except Exception:  # noqa: BLE001 - resolution is best-effort
        pass
    return None


def _outbox_chat_ref(chat: Any) -> Any:
    """A duck-typed chat ref for ``Outbox.submit`` — no social-layer
    import in the kernel. ChatRef-shaped objects pass through untouched;
    a ``"platform:chat_id"`` key string is decomposed the same way
    ``ChatRef.parse`` + ``ChatRef.key`` would rebuild it, so the outbox
    topic matches the gateway's own submits for the same chat."""
    if not isinstance(chat, str):
        return chat
    from types import SimpleNamespace

    _platform, _, rest = chat.partition(":")
    return SimpleNamespace(key=chat, chat_id=rest, kind="dm", thread_id="")


def _chat_platform(chat: Any) -> str:
    platform = getattr(chat, "platform", None)
    if platform:
        return str(platform)
    if isinstance(chat, str):
        head, sep, _ = chat.partition(":")
        if sep and head:
            return head
    return "telegram"


def drain_tg_retry_queue(context: Any, gateway: Any, *,
                         kv_factory: Callable[[Any], Any] | None) -> int:
    """Move legacy ``tg_retry:`` KV rows into the gateway's durable outbox.

    The telegram ladder's last rung queues into the KV store when no
    gateway exists (gateway-less processes: CLI one-shots). Those rows
    had NO drainer — the rung's "will be delivered" promise was false.
    The partner runtime calls this once its gateway exists: each row is
    submitted to the durable outbox (whose own drain machinery delivers
    it, in order, with receipts) and removed from the KV store. Returns
    the number of rows moved. Never raises.
    """
    moved = 0
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is None or gateway is None or kv_factory is None:
            return 0
        outbox = getattr(gateway, "outbox", None)
        if outbox is None:
            return 0
        kv = kv_factory(db)
        for key in kv.keys("tg_retry:"):
            try:
                row = kv.get(key)
                if not isinstance(row, dict):
                    kv.delete(key)
                    continue
                chat = str(row.get("chat") or "")
                text = str(row.get("text") or "")
                if not chat or not text:
                    # Undeliverable row — drop it, don't wedge the queue.
                    kv.delete(key)
                    continue
                outbox.submit(platform=_chat_platform(chat),
                              chat=_outbox_chat_ref(chat), text=text,
                              durable_kind="ladder")
                kv.delete(key)
                moved += 1
            except Exception:  # noqa: BLE001 - one bad row never wedges the rest
                continue
    except Exception:  # noqa: BLE001 - draining is best-effort at boot
        return moved
    if moved:
        _log.info("tg_retry drain: moved %d queued message(s) into the "
                  "durable outbox", moved)
    return moved


def register_default_ladders(
    es: ErrorSystem,
    context: Any = None,
    *,
    kv_factory: Callable[[Any], Any] | None = None,
    message_factory: Callable[[str], Any] | None = None,
    tts_factory: Callable[[Any, str], Any] | None = None,
    tts_probe: Callable[[], bool] | None = None,
) -> None:
    """Declare ladders for the real subsystems.

    Rung 0 is always the primary path; lower rungs degrade honestly.
    Handlers receive the same args the primary got, and resolve their
    resources (router / gateway / db) from ``context`` LAZILY at call
    time — the gateway in particular only exists once the partner
    runtime has started, after the error system is built.

    ``kv_factory`` builds the KV store for the telegram queue rung
    (``kv_factory(db)``); ``message_factory`` builds the LLM request
    messages for a prompt (``message_factory(prompt)``); ``tts_factory``
    is the voice layer's synthesis entry point
    (``tts_factory(context, text) -> audio path``) and ``tts_probe``
    its health check. All are injected by the caller — the kernel must
    not import storage, llm or voice. When a factory is None, the rung
    that needs it fails fast with an honest error instead of silently
    degrading.
    """
    journal = es.journal

    # ── LLM: provider chain as a ladder ──
    def _llm_primary(prompt: str, **kw: Any) -> Any:
        router = getattr(context, "router", None) if context else None
        if router is None:
            raise RuntimeError("no LLM router on context")
        if kw.get("messages"):
            msgs = kw["messages"]
        elif message_factory is not None:
            msgs = message_factory(prompt)
        else:
            raise RuntimeError(
                "no message factory injected — cannot build LLM request")
        # timeout_s is the CALLER's bound (the Brain runs its own
        # bounded thread); it is not a router.chat kwarg — forwarding
        # it would leak into provider payloads.
        kw.pop("timeout_s", None)
        result = router.chat(msgs, **{k: v for k, v in kw.items()
                                      if k != "messages"})
        # The real router REPORTS total failure as a response object
        # (LLMResponse.error), it does not raise. A failed response is
        # a rung failure: step down so the outage is journaled and the
        # honesty clause — not a bare provider error — is what surfaces.
        if result is not None and getattr(result, "ok", True) is False:
            raise RuntimeError(
                str(getattr(result, "error", "")
                    or "LLM router reported failure"))
        return result

    def _llm_probe_lazy() -> bool:
        # Resolves the router at probe time: the probe must work even
        # when the router was attached after registration (and must say
        # "not healthy" while there is no router at all).
        router = getattr(context, "router", None) if context else None
        if router is None:
            return False
        return _llm_probe(router)()

    def _llm_cached(prompt: str, **kw: Any) -> Any:
        raise RuntimeError(
            "all LLM providers unavailable and no cached answer for this prompt")

    es.ladders.register(DegradationLadder("llm", rungs=[
        Rung("primary", _llm_primary,
             probe=(_llm_probe_lazy if context is not None else None)),
        Rung("cached", _llm_cached,
             capability_trade="no live model; answers unavailable",
             honesty="All language models are unreachable right now — "
                     "I can't generate a fresh answer."),
    ], journal=journal))

    # ── Telegram: live send → durable queue ──
    def _tg_primary(chat: str, text: str, **kw: Any) -> Any:
        gw = _resolve_gateway(context)
        if gw is None:
            raise RuntimeError("no chat gateway on context")
        # durable=False on purpose: rung 0 is the LIVE attempt. A
        # durable rung 0 would park the message in the outbox before
        # the ladder has decided to degrade — and rung 1 would then
        # queue a second copy. The gateway's classified inner retry
        # still applies to the direct path.
        kw.pop("durable", None)
        result = gw.send("telegram", chat, text, durable=False, **kw)
        # gateway.send reports failure on the result, it does not
        # raise — a failed result is a rung failure, never a success.
        if result is not None and getattr(result, "ok", True) is False:
            raise RuntimeError(
                str(getattr(result, "error", "")
                    or "chat gateway reported send failure"))
        return result

    def _tg_queued(chat: str, text: str, **kw: Any) -> Any:
        # Last rung: park the message somewhere that MECHANICALLY
        # delivers it later. First choice: the gateway's durable
        # outbox (pass #38) — rows drain in order on later sends, with
        # receipts. Fallback (no gateway anywhere — CLI processes): the
        # legacy KV queue, whose rows the partner runtime drains into
        # the outbox at boot (drain_tg_retry_queue).
        gw = _resolve_gateway(context)
        outbox = getattr(gw, "outbox", None) if gw is not None else None
        if outbox is not None:
            try:
                job_id = outbox.submit(
                    platform=_chat_platform(chat),
                    chat=_outbox_chat_ref(chat), text=text,
                    durable_kind="ladder")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"could not queue telegram message: {exc}")
            return {"ok": False, "queued": True, "chat": chat,
                    "job_id": job_id}
        db = getattr(context, "db", None) if context else None
        if db is None:
            raise RuntimeError("no database to queue the message")
        if kv_factory is None:
            raise RuntimeError(
                "no KV store factory injected — cannot queue telegram message")
        try:
            kv_factory(db).set(f"tg_retry:{time.time()}:{chat}",
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

    # ── Voice: real TTS engine → silent (text) rung ──
    def _voice_primary(text: str, **kw: Any) -> Any:
        if tts_factory is None:
            raise RuntimeError(
                "voice primary rung has no TTS engine — no TTS factory "
                "was injected by the composition root")
        ctx = kw.get("context") or context
        if ctx is None:
            raise RuntimeError("voice rung has no context to synthesize with")
        path = tts_factory(ctx, text)
        if not path:
            raise RuntimeError("TTS engine returned no audio")
        return {"ok": True, "path": str(path), "silent": False}

    def _voice_silent(text: str, **kw: Any) -> Any:
        return {"ok": False, "silent": True,
                "note": "voice synthesis unavailable"}

    es.ladders.register(DegradationLadder("voice", rungs=[
        Rung("tts", _voice_primary, probe=tts_probe),
        Rung("silent", _voice_silent,
             capability_trade="no audio produced",
             honesty="Voice synthesis is unavailable right now — "
                     "here's the text instead."),
    ], journal=journal))

    _log.info("default degradation ladders registered: llm, telegram, voice")
