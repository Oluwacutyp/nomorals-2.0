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
  Same closure pattern. Builds ``ActiveSystems`` with real organ
  ADAPTERS injected into the L0 agency layer (this module is pinned
  L5, so it may import L1–L5) and runs ``work_shift()``. The
  research/wisdom/impossible organs are deliberately NOT constructed
  here: their ticks do heavyweight work (web research, corpus
  ingest, claim attempts) that a scheduled job should not silently
  start — those systems run organ-less, work their initiative
  lifecycles, and record honest "no organ injected" blockers.
  Scheduled shifts run under default-deny policy (observe/draft/propose, never silent
  outbound): the persistent ``MandateStore`` DOES apply to scheduled
  shifts, but ``Policy`` allow-rules do not persist, so silent
  posting needs the runtime's configured ``AgencyAuthority``, not a
  scheduled shift.
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
    """
    from ..core.policy import PermissionGradient, Policy

    policy = Policy()
    try:
        policy.load_rules(db)
    except Exception:  # noqa: BLE001 - unreadable rules = default-deny
        pass
    return PermissionGradient(policy)


def register(registry: Any) -> None:
    for name, spec in TOOLS.items():
        registry.register(
            name,
            spec["fn"],
            description=spec["description"],
            capability="owner",
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
    )
    def _agency_shift_tool() -> dict[str, Any]:
        # Real organs are constructed HERE (this module is pinned L5,
        # so it may import L1–L5) and INJECTED into the L0 agency
        # layer — the agency layer itself never imports organs
        # (ARCHITECTURE.md §1 layering). Each construction is
        # best-effort: a missing organ yields an organ-less adapter
        # (empty observations, refused actions), never a crash.
        import functools

        from ..agency.adapters import AgencyAuthority, build_adapters
        from ..agency.systems import ActiveSystems
        from ..autonomy.idle import workspace_db

        def _try(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - organ unavailable
                return None

        ws = getattr(getattr(_context, "settings", None),
                     "workspace_dir", None) or "."
        db = workspace_db(ws)

        def _news_agent():
            from ..agents.news import NewsAgent
            return NewsAgent(_context)

        def _social_manager():
            from ..social.manager import SocialManager
            return SocialManager(_context) if _context is not None else None

        def _gradient():
            return _gradient_for_shift(db)

        def _mandate_check():
            from ..finance.mandate import MandateStore, check_mandate
            return functools.partial(check_mandate, MandateStore())

        def _weather():
            from ..agents.weather import Weather
            return Weather()

        def _leads_store():
            from ..social.leads import LeadStore
            return LeadStore()

        def _opp_hunter():
            from ..agents.opportunities import OpportunityHunter
            return OpportunityHunter()

        def _applier():
            from ..agents.gig_applier import GigApplier
            return GigApplier()

        def _payout_extractor():
            from ..agents.opportunities import extract_payout_usd
            return extract_payout_usd

        def _opp_factory():
            from ..agents.opportunities import Opportunity
            return Opportunity.from_dict

        def _graph_fns():
            from ..planning.graph_missions import (
                PlanGraph, execute_graph, plan_authoring)
            return PlanGraph.from_dict, execute_graph, plan_authoring

        graph_factory, graph_execute, graph_compile = _try(_graph_fns) or (
            None, None, None)

        def _relationships():
            from ..social.relationships import RelationshipTracker
            return RelationshipTracker(
                path=str(ws) + "/data/social_relationships.json")

        def _draft_queue():
            from ..social.drafts import DraftQueue
            return DraftQueue()

        def _dm_sender(gateway):
            # The real DM-send channel: the chat gateway's send path.
            # None when no gateway is wired — sends then block honestly.
            if gateway is None:
                return None

            def _send(platform, contact, body):
                try:
                    res = gateway.send(platform, contact, body)
                    return bool(getattr(res, "ok", False))
                except Exception:  # noqa: BLE001 - send failure = not sent
                    return False

            return _send

        settings = getattr(_context, "settings", None)
        gateway = getattr(_context, "gateway", None)
        adapters = build_adapters(
            weather=_try(_weather), news_agent=_try(_news_agent),
            social_manager=_try(_social_manager),
            leads_store=_try(_leads_store),
            opp_hunter=_try(_opp_hunter), applier=_try(_applier),
            payout_extractor=_try(_payout_extractor),
            opp_factory=_try(_opp_factory),
            graph_factory=graph_factory, graph_execute_fn=graph_execute,
            graph_compile_fn=graph_compile, settings=settings,
            gateway=gateway, relationships=_try(_relationships),
            draft_queue=_try(_draft_queue), dm_sender=_dm_sender(gateway))
        authority = AgencyAuthority(
            gradient=_try(_gradient), grant=None,
            mandate_check=_try(_mandate_check))
        systems = ActiveSystems(db=db, adapters=adapters,
                                authority=authority)
        shift = systems.work_shift()
        # Expansion runs on the SAME schedule: SourceHealth actually
        # re-verifies sources every shift (the scheduler entrypoint).
        from ..agency.wiring import run_scheduled_expansion

        expansion = _try(
            lambda: run_scheduled_expansion(db, authority=authority))
        return {"ok": True, "shift": shift, "expansion": expansion}

    @registry.register(
        "__agent_modes_tick__",
        description="Run one autonomous-modes tick: every chat with the "
        "autonomous mode gets one self-initiation cycle (sources propose "
        "initiatives; ones needing approval land in /proposals). "
        "Scheduler-internal; runs every ~30 min.",
        capability="owner",
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
                from ..agents.notifier import Notifier

                # Pass 7 alert discipline: the event carries its
                # alert-policy classification (subject + hints); the
                # ONE policy in agents/alert_policy decides whether
                # this pages, digests, or logs silently.  Enforcement
                # itself already happened in the tick — only the
                # paging is tiered, never the response.
                Notifier(_context).publish(
                    "immune", event["title"], event["body"],
                    critical=bool(event.get("critical")),
                    alert_subject=str(event.get("subject") or ""),
                    alert_hints=event.get("hints"))
        except Exception:  # noqa: BLE001 - reporting never breaks the tick
            pass
        return result
