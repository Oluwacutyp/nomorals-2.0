"""The trigger engine: evaluates sources, fires actions, records history.

Design notes:

* **Schedule sources are not polled.**  They are wired into the existing
  :class:`nomorals.scheduler.scheduler.Scheduler` as real cron/one-time
  jobs whose action is ``__trigger_fire__``; the engine registers that
  action handler.  Reuse, not a parallel scheduler.
* **File/price/url sources are polled** by the engine's own thread (or by
  calling :meth:`tick` directly — tests and simple deployments do this).
  A trigger may set ``poll_s`` to override the engine's poll interval.
* **Message sources are event-driven**: :meth:`on_message` is called by
  the partner runtime's dispatch path (one hook call, no fork) and by
  anything else that wants to feed messages in.
* **Webhook sources are event-driven** via :meth:`fire_webhook`, served
  by ``triggers.webhook.register_trigger_routes`` on the API server.
  Schemes: plain shared secret, GitHub-style HMAC-SHA256, or
  Stripe-style ``t=…,v1=…`` with timestamp tolerance + idempotency-key
  replay dedup.
* **Bus sources are event-driven** via :meth:`attach_bus`: any event on
  the shared event bus (scheduler job finished, mission terminal,
  another trigger fired, …) is matched against ``bus``-source triggers.
  This is the cross-system wiring — systems wake each other through the
  bus instead of hoping.
* **Conditions** (HA-style gates: ``time_window`` / ``rate`` / ``evidence``)
  are evaluated after a source match and before the action; a failing
  gate records ``skipped`` with the reason — never a silent drop.
* **Modes**: ``parallel`` (default), ``single`` (skip while a run is in
  flight), ``queued`` (serialize runs).
* Resilience: one trigger's failing action is logged with the trigger
  id and recorded — it never kills the engine or other triggers.
  Disabled triggers never fire; the check happens at fire time, so a
  disable wins even against an already-queued scheduler job.
* No silent drops: every evaluation outcome (fired / no_match /
  skipped / error) is recorded in ``trigger_history``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.events import Event, global_bus
from ..storage.db import Database
from . import actions as _actions
from .models import (
    ACTION_MESSAGE,
    CONDITION_EVIDENCE,
    CONDITION_RATE,
    CONDITION_TIME_WINDOW,
    MODE_QUEUED,
    MODE_SINGLE,
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
    SKIP_ALREADY_RUNNING,
    SKIP_CONDITION,
    SKIP_COOLDOWN,
    SKIP_DISABLED,
    SOURCE_BUS,
    SOURCE_ENTITY_STATE,
    SOURCE_FILE,
    SOURCE_MESSAGE,
    SOURCE_PRICE,
    SOURCE_SCHEDULE,
    SOURCE_URL,
    SOURCE_WEBHOOK,
    WEBHOOK_SCHEME_GITHUB,
    WEBHOOK_SCHEME_STRIPE,
    Trigger,
    TriggerError,
    new_trigger_id,
    validate_trigger_spec,
)
from .sources import (
    SCHEDULER_ACTION,
    evaluate_file,
    evaluate_price,
    evaluate_url,
    match_bus,
    match_entity_state,
    match_message,
    schedule_plan,
)
from .store import TriggerStore

_log = get_logger(__name__)


def verify_webhook_signature(secret: str, scheme: str, *,
                             signature: str = "",
                             timestamp: str | int | float | None = None,
                             body: bytes = b"",
                             tolerance_s: float = 300,
                             now: float | None = None) -> None:
    """Verify an HMAC webhook signature (GitHub / Stripe parity).

    * ``github`` — ``signature`` is ``sha256=<hex(hmac_sha256(body))>``
      (``X-Hub-Signature-256`` style).
    * ``stripe`` — ``signature`` is ``t=<ts>,v1=<hex>`` where the hex is
      ``hmac_sha256(f"{ts}.{body}")``; the timestamp must be within
      ``tolerance_s`` of now (replay protection).

    Raises :class:`TriggerError` on any failure — missing signature,
    bad HMAC, or stale timestamp.  Uses :func:`hmac.compare_digest`
    (constant-time).

    Note: the API server route passes the parsed JSON body re-encoded;
    true raw-byte verification needs the raw request bytes — see
    ``triggers.webhook``.  For Devon's own webhooks the sender and this
    verifier must use the same canonical bytes.
    """
    import time as _time

    now = now if now is not None else _time.time()
    signature = str(signature or "")
    if scheme == WEBHOOK_SCHEME_GITHUB:
        if not signature:
            raise TriggerError("missing webhook signature")
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise TriggerError("bad webhook signature")
        return
    if scheme == WEBHOOK_SCHEME_STRIPE:
        parts: dict[str, str] = {}
        for piece in signature.split(","):
            if "=" in piece:
                k, v = piece.split("=", 1)
                parts[k.strip()] = v.strip()
        ts_raw = parts.get("t") or (str(timestamp)
                                    if timestamp is not None else "")
        v1 = parts.get("v1") or signature
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            raise TriggerError("bad webhook timestamp")
        if abs(now - ts) > tolerance_s:
            raise TriggerError(
                f"webhook timestamp outside tolerance ({tolerance_s:g}s)")
        signed = f"{ts_raw}.".encode() + body
        expected = hmac.new(secret.encode(), signed,
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, v1):
            raise TriggerError("bad webhook signature")
        return
    raise TriggerError(f"unknown webhook scheme {scheme!r}")


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break the trigger engine (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

#: context.extras key under which a live engine is published for the
#: partner runtime's message hook.
ENGINE_KEY = "trigger_engine"

_POLL_SOURCES = (SOURCE_FILE, SOURCE_PRICE, SOURCE_URL)


def _await(coro: Any) -> Any:
    """Run a scheduler coroutine from sync code.

    The scheduler's mutating API is async; the engine (like the
    scheduler's own worker) drives it with ``asyncio.run``.  Fail fast
    if a loop is already running — silently nesting loops is how
    deadlocks are born.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise TriggerError(
        "trigger schedule wiring cannot run inside a running event loop")


class TriggerEngine:
    """Evaluates triggers and fires their actions."""

    def __init__(
        self,
        db: Database,
        context: Any = None,
        *,
        scheduler: Any = None,
        resources: Any = None,
        send_message: Callable[[str, str], Any] | None = None,
        notify_fn: Callable[..., Any] | None = None,
        run_command: Callable[[list[str], float], dict[str, Any]] | None = None,
        start_mission: Callable[[str, int, Any], dict[str, Any]] | None = None,
        poll_interval: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.context = context
        self.store = TriggerStore(db)
        self._scheduler = scheduler
        #: Optional resource advisor (duck-typed: consult(mission) -> dict).
        #: Injected by L7 entry points; keeps L5 -> L6 imports out of this module.
        self._resources = resources
        self.send_message = send_message
        self.notify_fn = notify_fn
        self.run_command = run_command
        self.start_mission = start_mission
        self.poll_interval = max(1.0, float(poll_interval))
        self._clock = clock
        #: per-trigger source memory (file baselines, last prices, url digests)
        self._source_state: dict[str, dict[str, Any]] = {}
        #: per-trigger run locks for mode=single/queued
        self._run_locks: dict[str, threading.Lock] = {}
        #: evidence of the fire currently executing (read by actions for
        #: {{evidence}} template rendering)
        self._current_evidence: dict[str, Any] = {}
        #: per-trigger last poll time (backs per-trigger poll_s overrides)
        self._last_poll: dict[str, float] = {}
        #: bus attachment (attach_bus): the bus object + subscription id
        self._bus: Any = None
        self._bus_sub_id: str | None = None
        #: re-entrancy guard: a trigger firing publishes trigger.fired on
        #: the bus, which can legitimately chain into another bus trigger —
        #: cap the nesting so a cyclic chain degrades instead of recursing
        #: forever.
        self._bus_depth = 0
        self._running = False
        self._poll_thread: threading.Thread | None = None
        _log.info("TriggerEngine initialized")

    # ── event-bus source (cross-system triggering) ───────────────────────

    def attach_bus(self, bus: Any = None) -> str:
        """Subscribe to the shared event bus so ``bus``-source triggers
        fire on matching events.

        This is what makes the bus real instead of telemetry-only: a
        ``scheduler.job.finished`` event can start a mission, a
        ``mission.terminal`` event can notify, a ``trigger.fired`` event
        can chain another trigger.  The subscription is synchronous — the
        match-and-fire runs on the publisher's thread — so handlers stay
        on the engine's own fast, isolated ``_fire`` path.  Returns the
        subscription id.  Idempotent: attaching twice reuses the first.
        """
        if self._bus_sub_id is not None:
            return self._bus_sub_id
        if bus is None:
            from ..core.events import global_bus

            bus = global_bus
        self._bus = bus
        self._bus_sub_id = bus.subscribe("*", self.on_bus_event, sync=True)
        _log.info("TriggerEngine attached to event bus (sub %s)",
                  self._bus_sub_id)
        return self._bus_sub_id

    def detach_bus(self) -> None:
        """Remove the bus subscription.  Never raises."""
        try:
            if self._bus is not None and self._bus_sub_id is not None:
                self._bus.unsubscribe(self._bus_sub_id)
        except Exception:  # noqa: BLE001
            _log.debug("trigger bus detach failed", exc_info=True)
        finally:
            self._bus = None
            self._bus_sub_id = None

    def on_bus_event(self, event: Any) -> list[str]:
        """Feed one bus event to every enabled bus-source trigger.

        Mirrors :meth:`on_message`: per-trigger isolation, history for
        every outcome, never raises.  Chain depth is capped
        (:attr:`_bus_depth`) so cyclic trigger chains degrade loudly
        instead of recursing.
        """
        fired: list[str] = []
        if self._bus_depth >= 8:
            _log.warning("trigger bus chain depth exceeded — dropping event %s",
                         getattr(event, "topic", "?"))
            return fired
        self._bus_depth += 1
        try:
            for trigger in self.store.list(enabled_only=True,
                                           source=SOURCE_BUS):
                try:
                    hit, evidence = match_bus(trigger, event)
                except Exception as exc:  # noqa: BLE001 - per-trigger isolation
                    _log.exception("trigger %s bus match failed", trigger.id)
                    self.store.record(
                        trigger.id, OUTCOME_ERROR, {"source": "bus"},
                        error=f"{type(exc).__name__}: {exc}")
                    continue
                if hit:
                    result = self._fire(
                        trigger, {"source": "bus", **evidence})
                    if result["fired"]:
                        fired.append(trigger.id)
                else:
                    self.store.record(
                        trigger.id, OUTCOME_NO_MATCH,
                        {"source": "bus",
                         "topic": getattr(event, "topic", ""), **evidence})
        finally:
            self._bus_depth -= 1
        return fired

    # ── scheduler ────────────────────────────────────────────────────────

    def _ensure_scheduler(self) -> Any:
        if self._scheduler is None:
            from ..scheduler.scheduler import Scheduler

            self._scheduler = Scheduler(
                self.db, resources=self._resources)
        return self._scheduler

    @staticmethod
    def _job_id(trigger_id: str) -> str:
        return f"trigger:{trigger_id}"

    def _clear_scheduler_job(self, job_id: str) -> None:
        """Delete any scheduler rows for this job id (idempotent re-wire)."""
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM cron_jobs WHERE task_id = ?", (job_id,))
            self.db.execute(
                "DELETE FROM reminders WHERE task_id = ?", (job_id,))
            self.db.execute(
                "DELETE FROM event_hooks WHERE task_id = ?", (job_id,))
            self.db.execute(
                "DELETE FROM scheduled_tasks WHERE task_id = ?", (job_id,))

    def _wire_schedule(self, trigger: Trigger) -> None:
        """Wire a schedule trigger into the existing scheduler.

        The trigger's schedule options (``misfire`` / ``stale_after_s`` /
        ``overlap`` / ``timezone``) map straight onto the scheduler's own
        missed-fire and overlap policies — reuse, not a parallel
        scheduler.
        """
        sch = self._ensure_scheduler()
        sch.register_action(SCHEDULER_ACTION, self._on_scheduler_fire)
        job_id = self._job_id(trigger.id)
        self._clear_scheduler_job(job_id)
        kind, plan = schedule_plan(trigger.condition)
        cond = trigger.condition
        params = {"trigger_id": trigger.id}
        missed_fire_policy = cond.get("misfire", "fire_now")
        stale_after_s = float(cond.get("stale_after_s") or 86400)
        overlap_policy = cond.get("overlap", "skip")
        tz = cond.get("timezone") or None
        if kind == "cron":
            _await(sch.schedule_cron(
                job_id, plan["cron_expr"], SCHEDULER_ACTION, params,
                missed_fire_policy=missed_fire_policy,
                stale_after_s=stale_after_s,
                overlap_policy=overlap_policy,
                tz=tz))
        else:
            _await(sch.schedule_once(
                job_id, plan["run_at"], SCHEDULER_ACTION, params,
                missed_fire_policy=missed_fire_policy,
                stale_after_s=stale_after_s,
                overlap_policy=overlap_policy,
                tz=tz))
        _log.info("wired schedule trigger %s (%s %s)",
                  trigger.id, kind, plan)

    def _unwire_schedule(self, trigger_id: str) -> None:
        sch = self._ensure_scheduler()
        self._clear_scheduler_job(self._job_id(trigger_id))
        _log.info("unwired schedule trigger %s", trigger_id)

    async def _on_scheduler_fire(self, trigger_id: str) -> None:
        """The ``__trigger_fire__`` scheduler action handler."""
        self._fire_by_id(trigger_id, {"source": "schedule",
                                      "via": "scheduler"})

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler worker and the poll thread."""
        if self._running:
            return
        if getattr(self.db, "path", None) is None:
            raise TriggerError(
                "TriggerEngine.start() needs a file-backed Database — "
                "worker threads cannot share a :memory: database. Use "
                "tick() directly with :memory: instead.")
        sch = self._ensure_scheduler()
        sch.register_action(SCHEDULER_ACTION, self._on_scheduler_fire)
        sch.start()
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="trigger-poll")
        self._poll_thread.start()
        _log.info("TriggerEngine started")

    def stop(self) -> None:
        """Stop the poll thread and the scheduler worker."""
        self._running = False
        self.detach_bus()
        if self._scheduler is not None:
            try:
                self._scheduler.stop()
            except Exception:  # noqa: BLE001 - stop must not raise
                _log.exception("scheduler stop failed")
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)
            self._poll_thread = None
        _log.info("TriggerEngine stopped")

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad tick never kills the loop
                _log.exception("trigger poll tick failed")
            deadline = self._clock() + self.poll_interval
            while self._running and self._clock() < deadline:
                time.sleep(min(1.0, deadline - self._clock()))

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add(
        self,
        name: str,
        source: str,
        condition: dict[str, Any] | None,
        action: str,
        action_params: dict[str, Any] | None = None,
        *,
        cooldown_s: float = 0.0,
        enabled: bool = True,
        conditions: list[dict[str, Any]] | None = None,
        mode: str = "parallel",
        poll_s: float = 0.0,
    ) -> Trigger:
        """Validate (fail fast), persist, and wire a new trigger.

        ``conditions`` are HA-style gates evaluated after a source match
        (``time_window`` / ``rate`` / ``evidence``); ``mode`` is
        ``parallel`` | ``single`` | ``queued``; ``poll_s`` overrides the
        engine poll interval for this trigger's source.
        """
        if not str(name or "").strip():
            raise TriggerError("trigger needs a name")
        spec = validate_trigger_spec(
            source, condition, action, action_params, cooldown_s=cooldown_s,
            conditions=conditions, mode=mode, poll_s=poll_s)
        trigger = Trigger(
            id=new_trigger_id(), name=str(name).strip(), enabled=enabled,
            source=source, condition=spec.condition, action=action,
            action_params=spec.params, cooldown_s=spec.cooldown,
            conditions=spec.conditions, mode=spec.mode, poll_s=spec.poll_s)
        self.store.save(trigger)
        if enabled and source == SOURCE_SCHEDULE:
            try:
                self._wire_schedule(trigger)
            except Exception:
                # never leave a half-wired trigger behind
                self.store.delete(trigger.id)
                raise
        _log.info("added trigger %s (%s -> %s)", trigger.id, source, action)
        _emit("trigger.added", {
            "trigger_id": trigger.id,
            "name": trigger.name,
            "source": trigger.source,
            "action": trigger.action,
            "enabled": trigger.enabled,
            "mode": trigger.mode,
            "conditions": [c.get("type") for c in trigger.conditions],
        })
        return trigger

    def add_from_template(self, template_name: str,
                          params: dict[str, Any] | None = None,
                          **overrides: Any) -> Trigger:
        """Create a trigger from a built-in template (blueprint).

        ``params`` fills the template's ``{inputs}``; ``overrides`` set
        any trigger field (``enabled``, ``mode``, ``conditions`` …).
        """
        from .templates import render_template

        spec = render_template(template_name, params)
        name = str(overrides.pop("name", None) or spec.get("name")
                   or template_name)
        return self.add(
            name,
            str(spec.get("source") or "schedule"),
            dict(spec.get("condition") or {}),
            str(spec.get("action") or "notify"),
            dict(spec.get("action_params") or {}),
            cooldown_s=float(spec.get("cooldown_s") or 0.0),
            enabled=bool(overrides.pop("enabled", True)),
            conditions=overrides.pop("conditions", None),
            mode=str(overrides.pop("mode", "parallel")),
            poll_s=float(overrides.pop("poll_s",
                                       spec.get("poll_s") or 0.0)),
        )

    def remove(self, trigger_id: str) -> bool:
        trigger = self.store.get(trigger_id)
        if trigger is None:
            return False
        if trigger.source == SOURCE_SCHEDULE:
            try:
                self._unwire_schedule(trigger_id)
            except Exception:  # noqa: BLE001 - keep removing anyway
                _log.exception("unwire failed for %s", trigger_id)
        self._source_state.pop(trigger_id, None)
        removed = self.store.delete(trigger_id)
        if removed:
            _emit("trigger.removed", {
                "trigger_id": trigger_id,
                "name": trigger.name,
            })
        return removed

    def set_enabled(self, trigger_id: str, enabled: bool) -> Trigger:
        trigger = self.store.get(trigger_id)
        if trigger is None:
            raise TriggerError(f"unknown trigger {trigger_id!r}")
        self.store.set_enabled(trigger_id, enabled)
        trigger.enabled = enabled
        if trigger.source == SOURCE_SCHEDULE:
            if enabled:
                self._wire_schedule(trigger)
            else:
                self._unwire_schedule(trigger_id)
        _emit("trigger.enabled" if enabled else "trigger.disabled", {
            "trigger_id": trigger_id,
            "name": trigger.name,
        })
        return trigger

    def get(self, trigger_id: str) -> Trigger | None:
        return self.store.get(trigger_id)

    def list(self, *, enabled_only: bool = False,
             source: str | None = None) -> list[Trigger]:
        return self.store.list(enabled_only=enabled_only, source=source)

    def history(self, trigger_id: str | None = None,
                *, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.history(trigger_id, limit=limit)

    # ── evaluation: poll sources ──────────────────────────────────────────

    def _poll_due(self, trigger: Trigger, now: float) -> bool:
        """Per-trigger poll cadence override.  Only applies when the
        trigger sets ``poll_s`` explicitly — otherwise every tick
        evaluates (legacy behavior).  Not-due triggers are skipped
        silently (not an evaluation — nothing to record)."""
        if not trigger.poll_s:
            return True
        last = self._last_poll.get(trigger.id)
        if last is not None and now - last < trigger.poll_s:
            return False
        self._last_poll[trigger.id] = now
        return True

    def tick(self) -> dict[str, int]:
        """Evaluate every enabled file/price/url trigger once."""
        summary = {"evaluated": 0, "fired": 0, "errors": 0, "not_due": 0}
        now = self._clock()
        for trigger in self.store.list(enabled_only=True):
            if trigger.source == SOURCE_FILE:
                fn = evaluate_file
            elif trigger.source == SOURCE_PRICE:
                fn = evaluate_price
            elif trigger.source == SOURCE_URL:
                fn = evaluate_url
            else:
                continue
            if not self._poll_due(trigger, now):
                summary["not_due"] += 1
                continue
            summary["evaluated"] += 1
            state = self._source_state.setdefault(trigger.id, {})
            try:
                fired, evidence = fn(trigger, state)
            except Exception as exc:  # noqa: BLE001 - per-trigger isolation
                _log.exception("trigger %s evaluation failed", trigger.id)
                self.store.record(
                    trigger.id, OUTCOME_ERROR, {"source": trigger.source},
                    error=f"{type(exc).__name__}: {exc}")
                summary["errors"] += 1
                continue
            if fired:
                result = self._fire(
                    trigger, {"source": trigger.source, **evidence})
                if result["fired"]:
                    summary["fired"] += 1
                elif result["outcome"] == OUTCOME_ERROR:
                    summary["errors"] += 1
            else:
                self.store.record(
                    trigger.id, OUTCOME_NO_MATCH,
                    {"source": trigger.source, **evidence})
        try:
            self.store.purge_old()
            self.store.digest_purge()
        except Exception:  # noqa: BLE001 - purge is housekeeping, never fatal
            _log.exception("trigger history purge failed")
        try:
            flushed = self.flush_digests()
            summary["digests_flushed"] = flushed["sent"]
        except Exception:  # noqa: BLE001 - flush is housekeeping, never fatal
            _log.exception("trigger digest flush failed")
        return summary

    # ── evaluation: message source ─────────────────────────────────────────

    def on_message(self, text: str, chat_key: str, *,
                   platform: str = "", sender: str = "") -> list[str]:
        """Feed one inbound message to every enabled message trigger."""
        fired: list[str] = []
        for trigger in self.store.list(enabled_only=True,
                                       source=SOURCE_MESSAGE):
            try:
                hit, evidence = match_message(trigger, text, chat_key, sender)
            except Exception as exc:  # noqa: BLE001 - per-trigger isolation
                _log.exception("trigger %s message match failed", trigger.id)
                self.store.record(
                    trigger.id, OUTCOME_ERROR, {"source": "message"},
                    error=f"{type(exc).__name__}: {exc}")
                continue
            if hit:
                result = self._fire(
                    trigger, {"source": "message", "chat": chat_key,
                              "platform": platform, "sender": sender,
                              **evidence})
                if result["fired"]:
                    fired.append(trigger.id)
            else:
                self.store.record(
                    trigger.id, OUTCOME_NO_MATCH,
                    {"source": "message", "chat": chat_key, **evidence})
        return fired

    # ── evaluation: entity_state source ────────────────────────────────

    def on_entity_state(self, entity_id: str, old_state: Any,
                        new_state: Any,
                        attributes: dict[str, Any] | None = None
                        ) -> list[str]:
        """Feed one HA ``state_changed`` event to every enabled
        entity_state trigger.  Mirrors :meth:`on_message`: per-trigger
        isolation, history for every outcome, never raises."""
        fired: list[str] = []
        entity_id = str(entity_id or "").lower()
        attributes = dict(attributes or {})
        for trigger in self.store.list(enabled_only=True,
                                       source=SOURCE_ENTITY_STATE):
            try:
                hit, evidence = match_entity_state(
                    trigger, entity_id, old_state, new_state, attributes)
            except Exception as exc:  # noqa: BLE001 - per-trigger isolation
                _log.exception("trigger %s entity_state match failed",
                               trigger.id)
                self.store.record(
                    trigger.id, OUTCOME_ERROR, {"source": "entity_state"},
                    error=f"{type(exc).__name__}: {exc}")
                continue
            if hit:
                result = self._fire(
                    trigger, {"source": "entity_state",
                              "entity_id": entity_id,
                              "attributes": attributes,
                              **evidence})
                if result["fired"]:
                    fired.append(trigger.id)
            else:
                self.store.record(
                    trigger.id, OUTCOME_NO_MATCH,
                    {"source": "entity_state", "entity_id": entity_id,
                     **evidence})
        return fired

    # ── evaluation: webhook source ─────────────────────────────────────────

    def fire_webhook(self, trigger_id: str, *,
                     payload: dict[str, Any] | None = None,
                     secret: str | None = None,
                     signature: str = "",
                     timestamp: str | int | float | None = None,
                     event_id: str = "",
                     raw_body: bytes = b"") -> dict[str, Any]:
        """Fire a webhook trigger.  Fail fast on unknown id, wrong source,
        bad auth, replayed event id, or a disabled trigger — never
        silently drop.

        Auth schemes (set on the trigger condition): ``plain`` compares
        ``secret``; ``github`` / ``stripe`` verify an HMAC signature over
        the body (Stripe additionally enforces a timestamp tolerance).
        ``event_id`` is an idempotency key — a repeat inside the TTL is
        rejected as a replay (Stripe webhook_events parity).
        """
        import json as _json

        trigger = self.store.get(trigger_id)
        if trigger is None:
            raise TriggerError(f"unknown trigger {trigger_id!r}")
        if trigger.source != SOURCE_WEBHOOK:
            raise TriggerError(
                f"trigger {trigger_id!r} is not a webhook trigger "
                f"(source={trigger.source!r})")
        cond = trigger.condition
        scheme = cond.get("scheme") or "plain"
        if scheme == "plain":
            expected = cond.get("secret")
            if expected and not hmac.compare_digest(secret or "", expected):
                raise TriggerError("bad webhook secret")
        else:
            body = raw_body or _json.dumps(
                payload or {}, sort_keys=True,
                separators=(",", ":")).encode()
            verify_webhook_signature(
                str(cond.get("secret") or ""), scheme,
                signature=signature or str(secret or ""),
                timestamp=timestamp, body=body,
                tolerance_s=float(cond.get("tolerance_s") or 300))
        event_id = str(event_id or "")
        if event_id and self.store.seen_webhook_event(trigger_id, event_id):
            raise TriggerError(
                f"duplicate webhook event {event_id!r} (replay dedup)")
        if event_id:
            # note AFTER auth passes, so bad signatures can't pollute the
            # dedup table and lock out a later legitimate retry
            self.store.note_webhook_event(trigger_id, event_id)
        if not trigger.enabled:
            raise TriggerError(f"trigger {trigger_id!r} is disabled")
        return self._fire(trigger, {"source": "webhook",
                                    "payload": dict(payload or {}),
                                    "event_id": event_id})

    def manual_fire(self, trigger_id: str) -> dict[str, Any]:
        """Fire a trigger on demand (``nm trigger run``)."""
        trigger = self.store.get(trigger_id)
        if trigger is None:
            raise TriggerError(f"unknown trigger {trigger_id!r}")
        if not trigger.enabled:
            raise TriggerError(f"trigger {trigger_id!r} is disabled")
        return self._fire(trigger, {"source": "manual"})

    def next_run(self, trigger_id: str) -> float | None:
        """Next scheduled fire (epoch) for a schedule trigger; None when
        not a schedule trigger or nothing upcoming."""
        trigger = self.store.get(trigger_id)
        if trigger is None or trigger.source != SOURCE_SCHEDULE:
            return None
        cond = trigger.condition
        if "once" in cond:
            ts = float(cond["once"])
            return ts if ts > self._clock() else None
        from ..scheduler.scheduler import CronParser

        try:
            return CronParser.next_run(str(cond["cron"]),
                                       after=self._clock())
        except Exception:  # noqa: BLE001 - never break status views
            _log.debug("next_run failed for %s", trigger_id, exc_info=True)
            return None

    def status(self, trigger_id: str) -> dict[str, Any]:
        """Full status snapshot: definition + next run + digest backlog +
        outcome stats.  Powers the rich status view."""
        trigger = self.store.get(trigger_id)
        if trigger is None:
            raise TriggerError(f"unknown trigger {trigger_id!r}")
        out = trigger.to_dict()
        out["next_run"] = self.next_run(trigger_id)
        out["digest_pending"] = len(self.store.digest_pending(trigger_id))
        lock = self._run_locks.get(trigger_id)
        out["running"] = bool(lock is not None and lock.locked())
        out["stats"] = self.store.stats(trigger_id)
        return out

    def flush_digests(self, *, force: bool = False) -> dict[str, int]:
        """Send one combined message per trigger with a due digest buffer.

        A buffer flushes when its oldest line is older than
        ``digest_every_s`` or it reached ``digest_max`` lines (Huginn
        digest-agent parity).  Never raises for a broken send path — the
        lines stay buffered and the error is recorded.
        """
        summary = {"triggers": 0, "sent": 0, "errors": 0}
        now = self._clock()
        for trigger in self.store.list(enabled_only=True):
            pending = self.store.digest_pending(trigger.id)
            if not pending:
                continue
            params = trigger.action_params
            every = float(params.get("digest_every_s") or 3600)
            max_n = int(params.get("digest_max") or 25)
            oldest = self.store.digest_oldest(trigger.id) or now
            if not force and len(pending) < max_n and now - oldest < every:
                continue
            summary["triggers"] += 1
            title, body = _actions.format_digest(trigger, pending)
            try:
                if trigger.action == ACTION_MESSAGE:
                    sender = self.send_message
                    if sender is None:
                        raise TriggerError(
                            "digest flush needs send_message bound")
                    sender(str(trigger.action_params.get("chat") or ""),
                           f"{title}\n{body}")
                else:  # notify (owner channel) for everything else
                    fn = self.notify_fn or _actions.default_notify
                    fn(trigger, title, body, self)
                taken = self.store.digest_take(trigger.id)
                self.store.record(
                    trigger.id, OUTCOME_FIRED,
                    {"source": "digest", "lines": len(taken)}, fired=True)
                summary["sent"] += 1
                _log.info("trigger %s flushed digest (%d lines)",
                          trigger.id, len(taken))
            except Exception as exc:  # noqa: BLE001 - lines stay buffered
                _log.exception("trigger %s digest flush failed", trigger.id)
                self.store.record(
                    trigger.id, OUTCOME_ERROR, {"source": "digest"},
                    error=f"{type(exc).__name__}: {exc}")
                summary["errors"] += 1
        return summary

    # ── firing ─────────────────────────────────────────────────────────────

    def _fire_by_id(self, trigger_id: str,
                    evidence: dict[str, Any]) -> dict[str, Any]:
        trigger = self.store.get(trigger_id)
        if trigger is None:
            _log.warning("scheduler fired unknown trigger %s", trigger_id)
            return {"fired": False, "outcome": "unknown_trigger"}
        return self._fire(trigger, evidence)

    def check_conditions(self, trigger: Trigger,
                         evidence: dict[str, Any],
                         now: float | None = None) -> tuple[bool, str]:
        """Evaluate HA-style gates after a source match, before the action.

        Returns ``(True, "")`` when every gate passes, else ``(False,
        reason)``.  Never raises — a broken gate fails closed (skip).
        """
        now = now if now is not None else self._clock()
        try:
            for cond in trigger.conditions or []:
                ctype = cond.get("type")
                if ctype == CONDITION_TIME_WINDOW:
                    hhmm = time.strftime("%H:%M", time.localtime(now))
                    after, before = cond["after"], cond["before"]
                    if after <= before:
                        inside = after <= hhmm < before
                    else:  # spans midnight
                        inside = hhmm >= after or hhmm < before
                    if not inside:
                        return False, (
                            f"time_window {after}–{before} "
                            f"(now {hhmm})")
                elif ctype == CONDITION_RATE:
                    window_start = now - float(cond["window_s"])
                    n = self.store.count_outcome(
                        trigger.id, OUTCOME_FIRED, window_start)
                    if n >= int(cond["max"]):
                        return False, (
                            f"rate {n}/{cond['max']} per "
                            f"{float(cond['window_s']):g}s")
                elif ctype == CONDITION_EVIDENCE:
                    want = cond.get("match") or {}
                    for key, expected in want.items():
                        if evidence.get(key) != expected:
                            return False, (
                                f"evidence mismatch on {key!r}")
        except Exception as exc:  # noqa: BLE001 - gates fail closed
            _log.warning("trigger %s condition check failed: %s",
                         trigger.id, exc)
            return False, f"condition error: {exc}"
        return True, ""

    def _fire(self, trigger: Trigger,
              evidence: dict[str, Any]) -> dict[str, Any]:
        """Fire one trigger: checks, gates, action, history.  Never raises
        for action failures — they are logged (with the trigger id) and
        recorded; other triggers are unaffected."""
        now = self._clock()
        if not trigger.enabled:
            self.store.record(trigger.id, OUTCOME_SKIPPED,
                              {"reason": SKIP_DISABLED, **evidence})
            _log.info("trigger %s skipped (disabled)", trigger.id)
            return {"fired": False, "outcome": OUTCOME_SKIPPED,
                    "reason": SKIP_DISABLED}
        if (trigger.cooldown_s > 0 and trigger.last_fired
                and now - trigger.last_fired < trigger.cooldown_s):
            self.store.record(trigger.id, OUTCOME_SKIPPED,
                              {"reason": SKIP_COOLDOWN,
                               "cooldown_s": trigger.cooldown_s, **evidence})
            return {"fired": False, "outcome": OUTCOME_SKIPPED,
                    "reason": SKIP_COOLDOWN}
        gate_ok, gate_reason = self.check_conditions(trigger, evidence, now)
        if not gate_ok:
            self.store.record(trigger.id, OUTCOME_SKIPPED,
                              {"reason": SKIP_CONDITION,
                               "condition": gate_reason, **evidence})
            _log.info("trigger %s skipped (condition: %s)",
                      trigger.id, gate_reason)
            return {"fired": False, "outcome": OUTCOME_SKIPPED,
                    "reason": SKIP_CONDITION, "detail": gate_reason}
        handler = _actions.ACTION_HANDLERS.get(trigger.action)
        if handler is None:  # pragma: no cover - validated at add time
            raise TriggerError(f"unknown action {trigger.action!r}")
        # concurrency mode: single skips while in flight, queued serializes
        lock = self._run_locks.setdefault(trigger.id, threading.Lock())
        holding = False
        if trigger.mode == MODE_SINGLE:
            if not lock.acquire(blocking=False):
                self.store.record(
                    trigger.id, OUTCOME_SKIPPED,
                    {"reason": SKIP_ALREADY_RUNNING, **evidence})
                return {"fired": False, "outcome": OUTCOME_SKIPPED,
                        "reason": SKIP_ALREADY_RUNNING}
            holding = True
        elif trigger.mode == MODE_QUEUED:
            lock.acquire()
            holding = True
        prev_evidence = self._current_evidence
        self._current_evidence = dict(evidence)
        try:
            try:
                result = handler(trigger, self)
            except Exception as exc:  # noqa: BLE001 - engine resilience
                _log.exception("trigger %s (%s) action %s failed",
                               trigger.id, trigger.name, trigger.action)
                self.store.record(trigger.id, OUTCOME_ERROR, dict(evidence),
                                  error=f"{type(exc).__name__}: {exc}")
                self._ledger("error", trigger,
                             f"{trigger.name} action {trigger.action} failed",
                             ok=False,
                             learned=f"{type(exc).__name__}: {exc}"[:200])
                return {"fired": False, "outcome": OUTCOME_ERROR,
                        "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._current_evidence = prev_evidence
            if holding:
                lock.release()
        self.store.record(trigger.id, OUTCOME_FIRED,
                          {"evidence": dict(evidence),
                           "result": _jsonable(result)},
                          fired=True)
        _log.info("trigger %s (%s) fired action %s",
                  trigger.id, trigger.name, trigger.action)
        self._ledger("fired", trigger,
                     f"{trigger.name} fired action {trigger.action}",
                     metadata={"evidence": {k: v for k, v in evidence.items()
                                            if isinstance(v, (str, int, float,
                                                              bool))}})
        _emit("trigger.fired", {
            "trigger_id": trigger.id,
            "name": trigger.name,
            "action": trigger.action,
            "evidence": dict(evidence),
        })
        return {"fired": True, "outcome": OUTCOME_FIRED, "result": result}

    # ── autonomy ledger ──────────────────────────────────────────────────
    def _ledger(self, kind: str, trigger: Trigger, summary: str, *,
                ok: bool = True, learned: str = "",
                metadata: dict[str, Any] | None = None) -> None:
        """Journal a trigger event into the unified autonomy ledger
        (best-effort).  The ledger answers "what has the automation been
        doing" across scheduler, missions, triggers, and the pulse."""
        try:
            from ..agents.autonomy_ledger import record_ledger

            record_ledger(self.db, "trigger", kind, trigger.id, summary,
                          ok=ok, learned=learned,
                          metadata={"trigger": trigger.name,
                                    "source": trigger.source,
                                    **(metadata or {})})
        except Exception:  # noqa: BLE001 - ledger never breaks firing
            _log.debug("trigger ledger write failed", exc_info=True)


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe projection for history detail blobs."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)[:500]


# ── runtime integration ──────────────────────────────────────────────────

def attach(engine: TriggerEngine, context: Any) -> TriggerEngine:
    """Publish a live engine on the context for the message hook."""
    extras = getattr(context, "extras", None)
    if isinstance(extras, dict):
        extras[ENGINE_KEY] = engine
    else:  # exotic contexts without an extras dict
        setattr(context, ENGINE_KEY, engine)
    return engine


def message_hook(context: Any, text: str, chat_key: str, *,
                 platform: str = "", sender: str = "") -> list[str]:
    """Entry point for the partner runtime's dispatch path.

    Returns the ids of triggers that fired (empty when no engine is
    attached — a no-op, never an error).  The runtime calls this with a
    lazy import so ``triggers`` stays out of its import graph.
    """
    engine = None
    extras = getattr(context, "extras", None)
    if isinstance(extras, dict):
        engine = extras.get(ENGINE_KEY)
    if engine is None:
        engine = getattr(context, ENGINE_KEY, None)
    if engine is None:
        return []
    return engine.on_message(text, chat_key, platform=platform, sender=sender)
