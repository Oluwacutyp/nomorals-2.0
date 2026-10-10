"""The trigger engine: evaluates sources, fires actions, records history.

Design notes:

* **Schedule sources are not polled.**  They are wired into the existing
  :class:`nomorals.scheduler.scheduler.Scheduler` as real cron/one-time
  jobs whose action is ``__trigger_fire__``; the engine registers that
  action handler.  Reuse, not a parallel scheduler.
* **File/price sources are polled** by the engine's own thread (or by
  calling :meth:`tick` directly — tests and simple deployments do this).
* **Message sources are event-driven**: :meth:`on_message` is called by
  the partner runtime's dispatch path (one hook call, no fork) and by
  anything else that wants to feed messages in.
* **Webhook sources are event-driven** via :meth:`fire_webhook`, served
  by ``triggers.webhook.register_trigger_routes`` on the API server.
* **Bus sources are event-driven** via :meth:`attach_bus`: any event on
  the shared event bus (scheduler job finished, mission terminal,
  another trigger fired, …) is matched against ``bus``-source triggers.
  This is the cross-system wiring — systems wake each other through the
  bus instead of hoping.
* Resilience: one trigger's failing action is logged with the trigger
  id and recorded — it never kills the engine or other triggers.
  Disabled triggers never fire; the check happens at fire time, so a
  disable wins even against an already-queued scheduler job.
* No silent drops: every evaluation outcome (fired / no_match /
  skipped / error) is recorded in ``trigger_history``.
"""

from __future__ import annotations

import asyncio
import hmac
import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.events import Event, global_bus
from ..storage.db import Database
from . import actions as _actions
from .models import (
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
    SOURCE_BUS,
    SOURCE_ENTITY_STATE,
    SOURCE_FILE,
    SOURCE_MESSAGE,
    SOURCE_PRICE,
    SOURCE_SCHEDULE,
    SOURCE_WEBHOOK,
    Trigger,
    TriggerError,
    new_trigger_id,
    validate_definition,
)
from .sources import (
    SCHEDULER_ACTION,
    evaluate_file,
    evaluate_price,
    match_bus,
    match_entity_state,
    match_message,
    schedule_plan,
)
from .store import TriggerStore

_log = get_logger(__name__)


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

_POLL_SOURCES = (SOURCE_FILE, SOURCE_PRICE)


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
        #: per-trigger source memory (file baselines, last prices)
        self._source_state: dict[str, dict[str, Any]] = {}
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
        """Wire a schedule trigger into the existing scheduler."""
        sch = self._ensure_scheduler()
        sch.register_action(SCHEDULER_ACTION, self._on_scheduler_fire)
        job_id = self._job_id(trigger.id)
        self._clear_scheduler_job(job_id)
        kind, plan = schedule_plan(trigger.condition)
        params = {"trigger_id": trigger.id}
        if kind == "cron":
            _await(sch.schedule_cron(
                job_id, plan["cron_expr"], SCHEDULER_ACTION, params))
        else:
            _await(sch.schedule_once(
                job_id, plan["run_at"], SCHEDULER_ACTION, params))
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
    ) -> Trigger:
        """Validate (fail fast), persist, and wire a new trigger."""
        if not str(name or "").strip():
            raise TriggerError("trigger needs a name")
        condition, params, cooldown = validate_definition(
            source, condition, action, action_params, cooldown_s=cooldown_s)
        trigger = Trigger(
            id=new_trigger_id(), name=str(name).strip(), enabled=enabled,
            source=source, condition=condition, action=action,
            action_params=params, cooldown_s=cooldown)
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
        })
        return trigger

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

    def tick(self) -> dict[str, int]:
        """Evaluate every enabled file/price trigger once."""
        summary = {"evaluated": 0, "fired": 0, "errors": 0}
        for trigger in self.store.list(enabled_only=True):
            if trigger.source == SOURCE_FILE:
                fn = evaluate_file
            elif trigger.source == SOURCE_PRICE:
                fn = evaluate_price
            else:
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
        except Exception:  # noqa: BLE001 - purge is housekeeping, never fatal
            _log.exception("trigger history purge failed")
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
                     secret: str | None = None) -> dict[str, Any]:
        """Fire a webhook trigger.  Fail fast on unknown id, wrong source,
        bad secret, or a disabled trigger — never silently drop."""
        trigger = self.store.get(trigger_id)
        if trigger is None:
            raise TriggerError(f"unknown trigger {trigger_id!r}")
        if trigger.source != SOURCE_WEBHOOK:
            raise TriggerError(
                f"trigger {trigger_id!r} is not a webhook trigger "
                f"(source={trigger.source!r})")
        expected = trigger.condition.get("secret")
        if expected and not hmac.compare_digest(secret or "", expected):
            raise TriggerError("bad webhook secret")
        if not trigger.enabled:
            raise TriggerError(f"trigger {trigger_id!r} is disabled")
        return self._fire(trigger, {"source": "webhook",
                                    "payload": dict(payload or {})})

    def manual_fire(self, trigger_id: str) -> dict[str, Any]:
        """Fire a trigger on demand (``nm trigger run``)."""
        trigger = self.store.get(trigger_id)
        if trigger is None:
            raise TriggerError(f"unknown trigger {trigger_id!r}")
        if not trigger.enabled:
            raise TriggerError(f"trigger {trigger_id!r} is disabled")
        return self._fire(trigger, {"source": "manual"})

    # ── firing ─────────────────────────────────────────────────────────────

    def _fire_by_id(self, trigger_id: str,
                    evidence: dict[str, Any]) -> dict[str, Any]:
        trigger = self.store.get(trigger_id)
        if trigger is None:
            _log.warning("scheduler fired unknown trigger %s", trigger_id)
            return {"fired": False, "outcome": "unknown_trigger"}
        return self._fire(trigger, evidence)

    def _fire(self, trigger: Trigger,
              evidence: dict[str, Any]) -> dict[str, Any]:
        """Fire one trigger: checks, action, history.  Never raises for
        action failures — they are logged (with the trigger id) and
        recorded; other triggers are unaffected."""
        now = self._clock()
        if not trigger.enabled:
            self.store.record(trigger.id, OUTCOME_SKIPPED,
                              {"reason": "disabled", **evidence})
            _log.info("trigger %s skipped (disabled)", trigger.id)
            return {"fired": False, "outcome": OUTCOME_SKIPPED,
                    "reason": "disabled"}
        if (trigger.cooldown_s > 0 and trigger.last_fired
                and now - trigger.last_fired < trigger.cooldown_s):
            self.store.record(trigger.id, OUTCOME_SKIPPED,
                              {"reason": "cooldown",
                               "cooldown_s": trigger.cooldown_s, **evidence})
            return {"fired": False, "outcome": OUTCOME_SKIPPED,
                    "reason": "cooldown"}
        handler = _actions.ACTION_HANDLERS.get(trigger.action)
        if handler is None:  # pragma: no cover - validated at add time
            raise TriggerError(f"unknown action {trigger.action!r}")
        try:
            result = handler(trigger, self)
        except Exception as exc:  # noqa: BLE001 - engine resilience
            _log.exception("trigger %s (%s) action %s failed",
                           trigger.id, trigger.name, trigger.action)
            self.store.record(trigger.id, OUTCOME_ERROR, dict(evidence),
                              error=f"{type(exc).__name__}: {exc}")
            self._ledger("error", trigger,
                         f"{trigger.name} action {trigger.action} failed",
                         ok=False, learned=f"{type(exc).__name__}: {exc}"[:200])
            return {"fired": False, "outcome": OUTCOME_ERROR,
                    "error": f"{type(exc).__name__}: {exc}"}
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
