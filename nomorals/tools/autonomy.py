"""Autonomy tools — weakness management, presence, patterns.

Spine tools so the brain can inspect and manage the autonomous
nervous system through plain language:

* ``weakness_list`` — open weaknesses and proposals
* ``weakness_approve`` — owner approves a fix proposal
* ``weakness_dismiss`` — owner dismisses a weakness
* ``presence_sense`` — what the system knows right now
* ``interests`` — current interest model
* ``__presence_heartbeat__`` — scheduler-internal heartbeat (notice,
  prepare, surface). Registered via closure in ``register()``, NOT via
  the ``TOOLS`` dict: the registry calls fns with no ``context`` arg,
  and a ``context``-taking fn would fail with TypeError (the pre-existing
  ``presence_sense`` failure mode). See ``pulse``/``briefing``.
* ``__agency_shift__`` — scheduler-internal agency shift (observe →
  see what needs doing → do it → verify → follow through, no prompt).
  Same closure pattern. The shift's systems are built by the ONE
  agency composition root (``agency/wiring.compose_agency`` — the
  same root the interactive runtime boot uses), so a scheduled shift
  runs with the real gateway (resolved from ``context.extras``,
  where the runtime binds it), the live ``web_search``-backed
  opportunity hunter, and the research/wisdom/impossible organs
  joined to their initiative lifecycles. Per-system wiring state is
  reported in the shift result's ``composition`` readiness map.
  Scheduled shifts run under the durable policy: the persistent
  ``MandateStore`` and the persisted ``Policy`` allow-rules BOTH
  apply to scheduled shifts (loaded by the composition root); with
  no persisted rules, default-deny stands.
* ``__agent_modes_tick__`` — scheduler-internal agent-modes tick: one
  autonomous cycle for every chat running the autonomous mode.
  Same closure pattern. Reads the runtime's persistent ``AgentModes``
  from ``context.extras["agent_modes"]`` (a fresh engine would have
  no sources and no pending initiatives); falls back to a default
  engine when the runtime never booted one.  Initiatives that need
  approval land in ``/proposals`` via the runtime's proposal sink —
  the tool itself never sends anything.
* ``__immune_sampler__`` — scheduler-internal immune sampler: one
  live process sample into the persisted adaptive baseline
  (immune/wiring.py), consulting the adaptive cancer gate. Same
  closure pattern. The monitor is shared per store path, so the
  sampler, ``/immune`` chat and ``nm immune`` all see one baseline.
  Reporting: the job is a routine tick (silent at the scheduler);
  the tool itself publishes real immune events (enforcement,
  incident close, baseline ready, tick error) via the Notifier —
  non-event ticks publish nothing.
* ``__failure_loop__`` — scheduler-internal cognition failure loop:
  cluster the recorded failures (TrajectoryStore over the workspace
  DB), build a sandbox-checked fix proposal per cluster, file the
  proposals as FailureKB notes. Same closure pattern. Routine job
  (silent at the scheduler); proposals never auto-apply.
* ``__orchestration__`` — scheduler-internal community orchestration
  tick (AA-18): heartbeat this device into the fleet session
  registry, route via the device Router, dispatch due scheduled
  group posts through the social gateway. Same closure pattern.
  Routine job (silent at the scheduler).
* ``__season_tick__`` — scheduler-internal games season tick
  (AA-16): roll every group with a season board into the current
  quarter and pay closed seasons' prizes through the group economy.
  Same closure pattern. Routine job (silent at the scheduler);
  the summary lands in the scheduler journal only.
"""

from __future__ import annotations

from typing import Any


def _db(context: Any) -> Any:
    from ..storage.db import Database
    ws = getattr(getattr(context, "settings", None), "workspace_dir", ".")
    return Database(ws or ".")


def weakness_list(context: Any) -> dict[str, Any]:
    """List open weaknesses and fix proposals."""
    from ..autonomy.weakness import open_weaknesses
    db = _db(context)
    items = open_weaknesses(db)
    if not items:
        return {"ok": True, "weaknesses": [],
                "note": "no open weaknesses — the system is healthy"}
    return {"ok": True, "weaknesses": items}


def weakness_approve(context: Any, weakness_id: str) -> dict[str, Any]:
    """Approve a weakness fix proposal (owner only)."""
    from ..autonomy.weakness import approve_weakness
    db = _db(context)
    if approve_weakness(db, weakness_id):
        return {"ok": True,
                "note": f"proposal {weakness_id} approved — "
                        "ready for sandbox build/test"}
    return {"ok": False,
            "error": "not found or not in 'proposed' state"}


def weakness_dismiss(context: Any, weakness_id: str) -> dict[str, Any]:
    """Dismiss a weakness (owner only)."""
    from ..autonomy.weakness import ensure_schema
    db = _db(context)
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'dismissed' WHERE id = ?",
        (weakness_id,))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": bool(cur.rowcount)}


def presence_sense(context: Any) -> dict[str, Any]:
    """What the system knows right now: time, windows, interests, health."""
    from ..autonomy.presence import sense
    db = _db(context)
    return {"ok": True, "presence": sense(db)}


def interests(context: Any, limit: int = 10) -> dict[str, Any]:
    """Current interest model — what the owner cares about, with decay."""
    from ..autonomy.patterns import current_interests, rising_interests
    db = _db(context)
    return {
        "ok": True,
        "interests": current_interests(db, limit=limit),
        "rising": rising_interests(db),
    }


TOOLS = {
    "weakness_list": {
        "fn": weakness_list,
        "description": "List open system weaknesses and fix proposals.",
        "capabilities": {"owner"},
    },
    "weakness_approve": {
        "fn": weakness_approve,
        "description": "Approve a weakness fix proposal by id. Owner only.",
        "capabilities": {"owner"},
        "args": {"weakness_id": "the weakness id"},
    },
    "weakness_dismiss": {
        "fn": weakness_dismiss,
        "description": "Dismiss a weakness by id. Owner only.",
        "capabilities": {"owner"},
        "args": {"weakness_id": "the weakness id"},
    },
    "presence_sense": {
        "fn": presence_sense,
        "description": "What the system knows right now: time, predicted "
                       "active windows, interests, open weaknesses.",
        "capabilities": {"owner"},
    },
    "interests": {
        "fn": interests,
        "description": "The owner's interest model — topics with "
                       "time-decayed scores, plus rising interests.",
        "capabilities": {"owner"},
    },
}


def _gradient_for_shift(db: Any) -> Any:
    """Permission gradient for a SCHEDULED shift.

    Loads the DURABLE policy allow-rules from ``db`` (see
    ``Policy.save_rules`` / ``Policy.load_rules``), so a scheduled
    shift runs under the SAME authority as interactive use — not a
    weaker default-deny shadow. Missing/empty table = no persisted
    rules = default-deny stands (never a widened grant).

    The implementation lives in the ONE composition root
    (``agency/wiring.durable_permission_gradient``); this name is
    kept as the shift tool's historical entry point.
    """
    from ..agency.wiring import durable_permission_gradient

    return durable_permission_gradient(db)


def register(registry: Any) -> None:
    for name, spec in TOOLS.items():
        registry.register(
            name,
            spec["fn"],
            description=spec["description"],
            capability="owner",
            result_protocol="dict_ok",
        )
    # Scheduler-internal heartbeat tool. The ``presence-heartbeat`` job is
    # stored with payload_kind="tool" and payload
    # {"tool": "__presence_heartbeat__"}, so the tool MUST exist in the
    # tool registry — the scheduler's ``register_action`` path (see
    # ``ensure_heartbeat_job``) targets a different scheduler class and
    # never runs here, which is why the job failed with "unknown tool"
    # every 30 min. Closure over ``registry.context``: ``registry.call``
    # passes no ``context`` arg, so a ``context``-taking fn (the ``TOOLS``
    # dict pattern) would fail with TypeError. Same pattern as
    # ``pulse``/``briefing``. Uses the canonical ``workspace_db`` (not
    # ``Database(workspace_dir)``) so the heartbeat shares one state
    # store with the rest of autonomy.
    _context = registry.context

    @registry.register(
        "__presence_heartbeat__",
        description="Run the slow presence heartbeat: notice rising "
        "interests, prepare surfaces, surface one due item. "
        "Scheduler-internal; runs every ~30 min.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _presence_heartbeat_tool() -> dict[str, Any]:
        from ..autonomy.idle import workspace_db
        from ..autonomy.presence import heartbeat

        ws = getattr(getattr(_context, "settings", None),
                     "workspace_dir", None) or "."
        db = workspace_db(ws)
        return {"ok": True, "heartbeat": heartbeat(db)}

    @registry.register(
        "__agency_shift__",
        description="Run one agency worker shift: observe every system, "
        "fire proactive triggers, work them through the rotation, "
        "verify. Scheduler-internal; runs every ~30 min.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _agency_shift_tool() -> dict[str, Any]:
        # The shift's systems come from the ONE composition root
        # (agency/wiring.compose_agency — the same root the
        # interactive runtime boot uses): real organs constructed
        # there and INJECTED into the L0 agency layer (the agency
        # layer itself never imports organs — ARCHITECTURE.md §1
        # layering). Composition is best-effort per organ: a missing
        # organ yields an organ-less adapter (empty observations,
        # refused actions) plus a concrete readiness entry, never a
        # crash and never a silent degraded twin.
        from ..agency.systems import ActiveSystems
        from ..agency.wiring import compose_agency, run_scheduled_expansion
        from ..autonomy.idle import workspace_db

        def _try(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - expansion is best-effort
                return None

        ws = getattr(getattr(_context, "settings", None),
                     "workspace_dir", None) or "."
        db = workspace_db(ws)
        comp = compose_agency(_context, db=db)
        systems = ActiveSystems(
            db=db, adapters=comp["adapters"], authority=comp["authority"],
            research_organ=comp["research_organ"],
            wisdom_organ=comp["wisdom_organ"],
            impossible=comp["impossible"])
        shift = systems.work_shift()
        # Expansion runs on the SAME schedule: SourceHealth actually
        # re-verifies sources every shift (the scheduler entrypoint).
        expansion = _try(
            lambda: run_scheduled_expansion(
                db, authority=comp["authority"]))
        return {"ok": True, "shift": shift, "expansion": expansion,
                "composition": comp["readiness"]}

    @registry.register(
        "__agent_modes_tick__",
        description="Run one autonomous-modes tick: every chat with the "
        "autonomous mode gets one self-initiation cycle (sources propose "
        "initiatives; ones needing approval land in /proposals). "
        "Scheduler-internal; runs every ~30 min.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _agent_modes_tick_tool() -> dict[str, Any]:
        # The runtime's persistent engine (context.extras["agent_modes"])
        # — it owns the registered sources, the proposal sink, and the
        # pending initiatives.  A fresh engine here would tick nothing.
        # Fallback: a default engine still ticks chats whose mode set
        # (in the settings-resolved store) includes autonomous.
        from ..agents.modes import AgentModes

        modes = None
        try:
            modes = (_context.extras or {}).get("agent_modes")
        except Exception:  # noqa: BLE001 - extras are best-effort
            modes = None
        if modes is None:
            try:
                modes = AgentModes()
            except Exception as exc:  # noqa: BLE001 - never crash the tick
                return {"ok": False, "error": f"no agent modes: {exc}"}
        try:
            out = modes.tick_all({})
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False, "error": f"agent modes tick failed: {exc}"}
        total = sum(len(v) for v in out.values())
        proposed = sum(1 for v in out.values() for i in v
                       if i.status == "proposed")
        done = sum(1 for v in out.values() for i in v
                   if i.status == "done")
        return {"ok": True, "chats": sorted(out),
                "initiatives": total, "done": done, "proposed": proposed}

    @registry.register(
        "__immune_sampler__",
        description="Collect one live process sample into the immune "
        "adaptive baseline: persisted, recency-windowed, and "
        "quarantined (never learned) while an incident is active. "
        "Consults the adaptive cancer gate on each sample. "
        "Scheduler-internal; runs every ~1 min.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _immune_sampler_tool() -> dict[str, Any]:
        # The shared monitor resolves its store from the same
        # settings the runtime uses, so the sampler, /immune status
        # and `nm immune` all read ONE baseline. The tick itself
        # never raises (a failed collect is an honest no-sample).
        from ..immune import wiring

        monitor = wiring.shared_monitor(wiring.default_store_path(_context))
        try:
            result = monitor.tick()
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            result = {"ok": False,
                      "error": f"immune sampler tick failed: {exc}"}
        # Reporting contract (Pass 3): the sampler job is a routine
        # tick, so the scheduler's generic per-run alert stays silent
        # for it — real immune events are published HERE, by the
        # immune system itself, exactly once per event (an
        # enforcement, a fever/incident/baseline-ready announcement,
        # a tick error). Non-events — no live sample on this
        # platform, a routine learned sample, a quarantine under an
        # already-announced incident — publish NOTHING. The tick
        # result still returns to the scheduler, which records it as
        # the job's last_result and in the run ledger: silence is a
        # chat-surface rule, not deleted evidence.
        try:
            event = wiring.summarize_tick_event(
                result, monitor.drain_announcements())
            if event is not None:
                from ..agents.alert_templates import immune_event_doc
                from ..agents.notifier import Notifier
                from ..social.chat.present import body_text

                # Pass 7 alert discipline: the event carries its
                # alert-policy classification (subject + hints); the
                # ONE policy in agents/alert_policy decides whether
                # this pages, digests, or logs silently.  Enforcement
                # itself already happened in the tick — only the
                # paging is tiered, never the response.
                # Formatting F5: the same event also publishes as a
                # presentation Doc (its structured facts as labeled
                # rows) under the exact historical kind + title, so
                # the policy sees identical inputs.
                doc = immune_event_doc(event)
                Notifier(_context).publish(
                    "immune", event["title"], body_text(doc),
                    critical=bool(event.get("critical")),
                    alert_subject=str(event.get("subject") or ""),
                    alert_hints=event.get("hints"),
                    doc=doc)
        except Exception:  # noqa: BLE001 - reporting never breaks the tick
            pass
        # Organ-bus emission (AA-42): an enforcement is a contained
        # threat — persist it as an immune event on the cross-organ
        # bus (dst "brain", like research/wisdom's decision-relevant
        # events) so other organs can react to it, not only the
        # owner's notification feed. Best-effort, like all reporting
        # here: a bus failure never breaks the tick.
        try:
            if result.get("enforced"):
                from .. import organs as _organs
                from ..immune.events import emit_immune_event

                def _bus_sink(kind: str, payload: dict) -> None:
                    bus_db = getattr(_context, "db", None) or _db(_context)
                    _organs.emit(bus_db, "immune", "brain", kind, payload)

                emit_immune_event(_bus_sink, "threat_detected", {
                    "source": "immune-sampler",
                    "sample": result.get("sample") or {},
                    "reason": result.get("reason") or "",
                })
                emit_immune_event(_bus_sink, "threat_contained", {
                    "source": "immune-sampler",
                    "action": "adaptive gate enforced — incident opened, "
                              "sample quarantined",
                    "reason": result.get("reason") or "",
                })
        except Exception:  # noqa: BLE001 - reporting never breaks the tick
            pass
        return result

    @registry.register(
        "__failure_loop__",
        description="Run one cognition failure-loop pass: cluster the "
        "recorded failures, build a gated fix proposal per cluster, "
        "sandbox-check each, and file the proposals as FailureKB "
        "notes. Proposals never auto-apply — approval stays the "
        "owner's. Scheduler-internal; runs every ~1h.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _failure_loop_tool() -> dict[str, Any]:
        # The loop reads the SAME DB the router's learning attachment
        # records trajectories into: agents/context wires
        # attach_learning(router, db=context.db) with the very object
        # the tool registry's context carries as .db — so the clusters
        # are the system's real recorded failures, not a side store.
        from ..cognition.failure_loop import run_failure_loop_once

        try:
            db = getattr(_context, "db", None) or _db(_context)
            return run_failure_loop_once(db)
        except Exception as exc:  # noqa: BLE001 - one bad pass never kills the job
            return {"ok": False,
                    "error": f"failure loop pass failed: {exc}"}

    @registry.register(
        "__season_tick__",
        description="Run one games season tick (AA-16): roll every "
        "group with a season board into the current quarter and pay "
        "closed seasons' prizes (ranks 1-3) through the group "
        "economy. Scheduler-internal; runs every ~1h.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _season_tick_tool() -> dict[str, Any]:
        from ..games.tournaments_seasons import run_season_tick_once

        try:
            return run_season_tick_once(context=_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"season tick failed: {exc}"}

    @registry.register(
        "__orchestration__",
        description="Run one community orchestration tick (AA-18): "
        "heartbeat this device into the fleet session registry, let "
        "the Router pick the device that should speak, and dispatch "
        "due scheduled group posts through the social gateway. "
        "Scheduler-internal; runs every ~1 min.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _orchestration_tool() -> dict[str, Any]:
        from ..community.orchestration import run_orchestration_tick

        def sender(platform: str, target: str, text: str) -> bool:
            from ..social.chat.base import ChatRef
            from .social import get_gateway

            gateway = get_gateway()
            if gateway is None:
                return False  # no gateway bound: plan stays unsent, honest
            if ":" in target:
                plat, _, chat_id = target.partition(":")
                platform, target = plat or platform, chat_id
            result = gateway.send(
                platform, ChatRef(platform=platform, chat_id=target,
                                  kind="group"), text)
            return bool(getattr(result, "ok", False))

        try:
            summary = run_orchestration_tick(sender, context=_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"orchestration tick failed: {exc}"}
        return {"ok": True, **summary}
