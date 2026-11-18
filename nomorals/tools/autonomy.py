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
  from ``context.extras["agent_modes"]``; falls back to a default
  engine wired the same way (proactive bridge + a proposal sink
  filing into the workspace DB) when the runtime never booted one.
  Initiatives that need approval land in ``/proposals`` via the
  runtime's proposal sink — the tool itself never sends anything.
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
* ``__group_backup_tick__`` — scheduler-internal group backup tick
  (§9.8): back up every group whose backup cadence is due.
  Same closure pattern. Routine job (silent at the scheduler);
  each run journals to its group's audit log.
* ``__group_roster_diff_tick__`` — scheduler-internal roster-diff
  tick (§9.1): diff every claimed group whose roster cadence is
  due against its live member list. Same closure pattern; the
  tool resolves the live gateway and hands the organ the
  adapters.
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


def _show_pulse_sections(context: Any) -> list[dict[str, Any]]:
    """The morning pulse as show segments: the latest STORED briefing's
    sections (read-only — the tick never re-renders or re-sends a
    briefing). The explore-trends section is skipped here because the
    research reader covers the same claims fresher. Empty when no
    briefing has ever been stored. Never raises.
    """
    try:
        from ..agents.morning_briefing import latest_briefing
    except Exception:  # noqa: BLE001 - no briefing module, no pulse
        return []
    try:
        briefing = latest_briefing(context)
    except Exception:  # noqa: BLE001 - unreadable briefing, no pulse
        return []
    if not briefing:
        return []
    out: list[dict[str, Any]] = []
    for section in (briefing.get("sections") or []):
        if not isinstance(section, dict):
            continue
        if str(section.get("name") or "") == "explore_trends":
            continue
        lines = [str(line) for line in (section.get("lines") or [])
                 if str(line).strip()]
        if not lines:
            continue
        title = str(section.get("title") or section.get("name")
                    or "Briefing")
        out.append({"title": title,
                    "text": f"{title}: " + " ".join(lines)})
        if len(out) >= 8:
            break
    return out


def _show_research_sections(context: Any) -> list[dict[str, Any]]:
    """Fresh exploration claims as show segments — the same read the
    briefing's ExploreTrendsProvider performs (shared explore store,
    CLAIM_TREND, standing fresh, observed in the last 24h). Empty
    when the field is quiet. Never raises.
    """
    try:
        from ..agents.morning_briefing import _explore_claim_line
        from ..media.explore import default_explore_store
        from ..media.explore.trends import CLAIM_TREND, current_claims
    except Exception:  # noqa: BLE001 - no explore store, no research
        return []
    try:
        import time as _time

        store = default_explore_store()
        db = getattr(context, "db", None)
        if db is not None:
            from ..storage.kv import KVStore

            store.bind_store(KVStore(db))
        now = _time.time()
        fresh = [c for c in current_claims(store, now=now)
                 if c.get("kind") == CLAIM_TREND
                 and c.get("standing") == "fresh"
                 and float(c.get("observed_at") or 0) >= now - 86400.0]
        return [{"title": "Research",
                 "text": _explore_claim_line(claim)}
                for claim in fresh[:3]]
    except Exception:  # noqa: BLE001 - unreadable claims, no research
        return []


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
        # pending initiatives.  Fallback: a default engine wired the
        # same way the runtime boot wires its own (proactive bridge +
        # a proposal sink filing into the workspace DB), so it still
        # ticks chats whose mode set (in the settings-resolved store)
        # includes autonomous — a bare engine would tick nothing.
        from ..agents.modes import (
            AgentModes,
            _home_from_settings,
            file_initiative_proposal,
            with_production_sources,
        )
        from ..autonomy.idle import workspace_db

        modes = None
        try:
            modes = (_context.extras or {}).get("agent_modes")
        except Exception:  # noqa: BLE001 - extras are best-effort
            modes = None
        if modes is None:
            try:
                ws = getattr(getattr(_context, "settings", None),
                             "workspace_dir", None) or "."
                db = workspace_db(ws)
                # The home is the settings home, NOT the workspace
                # dir (``<home>/workspace``): the engine must learn
                # into the same <home>/data store the boot uses.
                modes = with_production_sources(
                    AgentModes(),
                    proposal_sink=lambda init, spec, ctx: (
                        file_initiative_proposal(db, init, spec, ctx)),
                    home=_home_from_settings(
                        getattr(_context, "settings", None)))
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
        "__wal_checkpoint__",
        description="Run one WAL maintenance pass on the workspace DB: "
        "PASSIVE checkpoint first (never blocks — a reader holding a "
        "snapshot reports busy at once), TRUNCATE only when nothing "
        "pins the log, rewinding the -wal file to zero bytes. The WAL "
        "otherwise grows to its 64MB cap while long reads hold "
        "snapshots, and every new connection pays the walk. "
        "Scheduler-internal; runs every ~1h. Routine and silent.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _wal_checkpoint_tool() -> dict[str, Any]:
        try:
            db = getattr(_context, "db", None) or _db(_context)
            report = db.checkpoint_maintenance()
            return {"ok": True, **report}
        except Exception as exc:  # noqa: BLE001 - one bad pass never kills the job
            return {"ok": False,
                    "error": f"wal checkpoint pass failed: {exc}"}

    @registry.register(
        "__calibration_loop__",
        description="Run one calibration -> abstention -> learning-"
        "targets pass: replay trajectory rows into a fresh "
        "calibration store, rank learning targets and file the worst "
        "as model weaknesses, and persist per-slice abstention "
        "thresholds that move with measured overconfidence. "
        "Scheduler-internal; runs every ~1h. Routine and silent — "
        "thresholds are advisory values the abstention gate reads, "
        "targets surface in the weakness ledger; nothing auto-applies.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _calibration_loop_tool() -> dict[str, Any]:
        # The loop reads the SAME DB the router's learning attachment
        # records trajectories into: agents/context wires
        # attach_learning(router, db=context.db) with the very object
        # the tool registry's context carries as .db — so the replay
        # is the system's real recorded outcomes, not a side store.
        # The ranking call lives here (L5), not in the L3 loop
        # module: cognition must not import the autonomy organ.
        from ..autonomy.model_weakness import (
            learning_targets,
            record_model_weakness,
        )
        from ..cognition.calibration_loop import run_calibration_loop_once

        try:
            db = getattr(_context, "db", None) or _db(_context)
            result = run_calibration_loop_once(db)
            targets = learning_targets(result["store"])
            filed: list[str] = []
            errors = 0
            for target in targets:
                try:
                    filed.append(str(record_model_weakness(
                        db, task=target["task"], model=target["model"],
                        evidence=dict(target))))
                except Exception:  # noqa: BLE001 - one bad filing never kills the tick
                    errors += 1
            result["targets"] = targets
            result["weaknesses_filed"] = filed
            result["weakness_errors"] = errors
            # The rebuilt store is the tool's working object, not a
            # reportable value — the scheduler renders the result for
            # the owner, so it leaves the dict here.
            result.pop("store", None)
            return result
        except Exception as exc:  # noqa: BLE001 - one bad pass never kills the job
            return {"ok": False,
                    "error": f"calibration loop pass failed: {exc}"}

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
        "__group_backup_tick__",
        description="Run one group backup tick (§9.8): back up "
        "every group whose backup cadence is due (daily/weekly at "
        "its configured hour). Scheduler-internal; runs every ~1h. "
        "Routine and silent — each run journals to its group's "
        "audit log and never posts anywhere.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _group_backup_tick_tool() -> dict[str, Any]:
        from ..social.chat.group_backup import (
            run_group_backup_tick_once,
        )

        try:
            return run_group_backup_tick_once(context=_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"group backup tick failed: {exc}"}

    @registry.register(
        "__group_roster_diff_tick__",
        description="Run one group roster-diff tick (§9.1): diff "
        "every claimed group whose roster cadence is due against "
        "its live member list (userbot; Bot API probe fallback). "
        "Scheduler-internal; runs every ~1h. Routine and silent — "
        "each run journals to its group's audit log and never "
        "posts anywhere.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _group_roster_diff_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import resolve_gateway
        from ..social.chat.group_roster_diff import (
            run_roster_diff_tick_once,
        )

        try:
            gateway = resolve_gateway(_context)
            adapters = list(
                (getattr(gateway, "adapters", None) or {}).items()
            ) if gateway is not None else []
            return run_roster_diff_tick_once(
                context=_context, adapters=adapters)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"group roster diff tick failed: {exc}"}

    @registry.register(
        "__group_rhythm_tick__",
        description="Run one group rhythm tick (owner reports "
        "2026-11-16): run the rhythm + estate cycle for every "
        "claimed group and post due announcements — richest-"
        "balance records, district events, the weekly gift "
        "proposal — on their own clock, no message required. "
        "Bot-first: an owner-surface runtime defers to the bot's "
        "tick marker. Scheduler-internal; runs every ~5m. "
        "Routine and silent — silent hours hold notices into "
        "the morning digest, each run journals to its group's "
        "audit log.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _group_rhythm_tick_tool() -> dict[str, Any]:
        from ..agents.group_rhythm import run_group_rhythm_tick_once
        from ..agents.notifier import resolve_gateway

        try:
            gateway = resolve_gateway(_context)
            adapters = list(
                (getattr(gateway, "adapters", None) or {}).items()
            ) if gateway is not None else []
            return run_group_rhythm_tick_once(
                context=_context, adapters=adapters)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"group rhythm tick failed: {exc}"}

    @registry.register(
        "__group_digest_tick__",
        description="Run one group digest tick (audit gap R2-20): "
        "every claimed WhatsApp group with new messages since its "
        "watermark gets its digest — mentions, decisions, resolved "
        "threads — published to the owner via the Notifier. "
        "Scheduler-internal; runs every ~6h. Routine and silent — "
        "a second run with no new messages publishes nothing.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _group_digest_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..social.chat.whatsapp_digest import run_group_digest_tick_once

        try:
            return run_group_digest_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"group digest tick failed: {exc}"}

    @registry.register(
        "__weekly_review_tick__",
        description="Run one weekly review tick (small remainders "
        "bundle): build the week's review from the real ledgers "
        "(commitments, decisions, focus) and publish it to the owner "
        "via the Notifier, once per week (watermarked) — skipping "
        "empty weeks. Scheduler-internal; runs Sundays 20:00 local. "
        "Routine and silent — a second run in the same week publishes "
        "nothing.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _weekly_review_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..reviews.jobs import run_weekly_review_tick_once

        try:
            return run_weekly_review_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"weekly review tick failed: {exc}"}

    @registry.register(
        "__av_sweep_tick__",
        description="Run one antivirus sweep tick (AV wiring bundle): "
        "scan the media dir, quarantine hits, publish through the "
        "Notifier only when something was found (infected or unknown "
        "files), and write durable last-sweep state. Scheduler-"
        "internal; runs daily 03:00 local. Routine and silent — a "
        "clean sweep publishes nothing.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _av_sweep_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..security.av_sweep import run_av_sweep_tick_once

        try:
            return run_av_sweep_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"av sweep tick failed: {exc}"}

    @registry.register(
        "__breach_watch_tick__",
        description="Run one breach-watch tick (52-websites #23): check "
        "every watched address against HaveIBeenPwned and publish "
        "through the Notifier exactly once when a NEW breach names "
        "one. Scheduler-internal; runs daily 05:00 local. Routine "
        "and silent — a clean day, or a day with nothing new, "
        "publishes nothing. Without an HIBP key the tick reports "
        "ok False and publishes nothing.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _breach_watch_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..security.breach import run_breach_watch_tick_once

        try:
            return run_breach_watch_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"breach watch tick failed: {exc}"}

    @registry.register(
        "__airport_refresh_tick__",
        description="Run one airport dataset refresh tick (R3-67 "
        "remainder): fetch the canonical OurAirports dump, rebuild "
        "the IATA→country mapping with the generator's rules, "
        "validate it, atomically replace the table, and invalidate "
        "the module cache — a failed fetch or validation leaves the "
        "old table in place. Scheduler-internal; runs monthly on the "
        "1st at 04:00 local. Routine and silent on success; a "
        "failure is published to the owner.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _airport_refresh_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..travel.refresh import (
            COORDS_PATH,
            run_airport_refresh_tick_once,
        )

        try:
            return run_airport_refresh_tick_once(
                context=_context, notifier=Notifier(_context),
                coords_path=COORDS_PATH)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"airport refresh tick failed: {exc}"}

    @registry.register(
        "__flight_track_tick__",
        description="Run one flight-watch tick (FlightRadar24 "
        "build-new): check every armed flight watch against the "
        "live ADS-B feed and publish one Notifier note per watch "
        "that landed, diverted, ran late or vanished — one alert "
        "per event, never twice. Scheduler-internal; runs every "
        "15 minutes. Routine and silent when nothing changed; a "
        "feed failure reports the reason and never kills the job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _flight_track_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..travel.tracking import run_flight_track_tick_once

        try:
            return run_flight_track_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"flight track tick failed: {exc}"}

    @registry.register(
        "__price_history_tick__",
        description="Run one price-history tick (CamelCamelCamel "
        "extend): check every armed retail price watch against "
        "the live shopping engine and publish one Notifier note "
        "per watch that dropped below target (or a new low) or "
        "came back in stock — one alert per drop level, never "
        "twice. Scheduler-internal; runs daily at 07:00. Routine "
        "and silent when nothing changed; a quiet tick (no "
        "watches) makes no network call at all; a fetch failure "
        "is reported, never a dead job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _price_history_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..commerce.price_history import run_price_history_tick_once
        from .price_history import build_fetcher

        try:
            try:
                fetcher = build_fetcher()
            except Exception:  # noqa: BLE001 - tick reports, never raises
                fetcher = None
            return run_price_history_tick_once(
                context=_context, notifier=Notifier(_context),
                fetcher=fetcher)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"price history tick failed: {exc}"}

    @registry.register(
        "__snapshot_prune_tick__",
        description="Run one snapshot prune tick (Archive.ph "
        "extend): delete page snapshots older than the owner's "
        "retention setting — files and record together — and report "
        "what went. Scheduler-internal; runs daily at 03:00. "
        "Routine and silent when nothing is due; a tick with no "
        "snapshots makes no network call at all.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _snapshot_prune_tick_tool() -> dict[str, Any]:
        from ..browser.snapshot import SnapshotStore, prune

        try:
            settings = getattr(_context, "settings", None)
            home = getattr(settings, "home", None) if settings else None
            store = SnapshotStore(home) if home else SnapshotStore()
            return prune(store)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"snapshot prune tick failed: {exc}"}

    @registry.register(
        "__watchtower_tick__",
        description="Run one watchtower tick (Downdetector "
        "build-new): probe every watched external service and "
        "page the owner once when a critical target goes down — "
        "one alert per incident, never twice. Scheduler-internal; "
        "runs every 4 minutes. Routine and silent when nothing "
        "changed; a quiet tick (no targets) makes no network call "
        "at all; a probe failure is a sample, never a dead job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _watchtower_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..sources.watchtower import run_watchtower_tick_once

        try:
            return run_watchtower_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"watchtower tick failed: {exc}"}

    @registry.register(
        "__offer_reconcile_tick__",
        description="Run one offer-reconcile tick (JustWatch "
        "build-new): re-check every watchlisted film/show against "
        "the streaming catalogues and publish one Notifier note per "
        "unseen change (a new service picking the title up, or one "
        "dropping it) — one alert per event, never twice. "
        "Scheduler-internal; runs daily at 06:00. Routine and "
        "silent when nothing changed; a quiet tick (empty watchlist) "
        "makes no network call at all.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _offer_reconcile_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..media.film_sources import run_offer_reconcile_tick_once

        try:
            return run_offer_reconcile_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"offer reconcile tick failed: {exc}"}

    @registry.register(
        "__burn_note_tick__",
        description="Run one burn-note sweep (Privnote build-new): "
        "delete expired unread burn notes from the durable store "
        "and log each burn with its reason. Scheduler-internal; "
        "runs hourly. Routine and silent — a run with nothing "
        "expired deletes nothing, and a live note is never touched.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _burn_note_tick_tool() -> dict[str, Any]:
        from ..memory.ephemeral_notes import run_burn_note_tick_once

        try:
            return run_burn_note_tick_once(context=_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"burn note tick failed: {exc}"}

    @registry.register(
        "__own_domain_mail_tick__",
        description="Run one own-domain mail sweep (10MinuteMail "
        "build): delete expired own-domain mailboxes and their "
        "messages from the durable store — deleted, not archived — "
        "and report live counts. Scheduler-internal; runs every 5 "
        "minutes. Routine and silent — a run with nothing expired "
        "deletes nothing, and a live mailbox is never touched.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _own_domain_mail_tick_tool() -> dict[str, Any]:
        from ..accounts.temp_mail import run_own_domain_mail_tick_once

        try:
            return run_own_domain_mail_tick_once(context=_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"own-domain mail tick failed: {exc}"}

    @registry.register(
        "__share_tick__",
        description="Run one share sweep (File.io build): delete "
        "expired and exhausted file shares from the durable store — "
        "record and blob ref both — and journal each death with its "
        "reason. Scheduler-internal; runs hourly. Routine and silent "
        "— a run with nothing expired deletes nothing, and a live "
        "share is never touched.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _share_tick_tool() -> dict[str, Any]:
        from ..storage.share import run_share_tick_once

        try:
            return run_share_tick_once(context=_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"share sweep failed: {exc}"}

    @registry.register(
        "__paper_feed_tick__",
        description="Run one paper-feed tick (Semantic Scholar "
        "paper records): check every watched topic against Semantic "
        "Scholar search and publish one Notifier note per topic with "
        "new papers — a paper is reported exactly once. "
        "Scheduler-internal; runs daily 08:00 local. Routine and "
        "silent when nothing is watched or nothing is new.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _paper_feed_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..research.paper_records import run_paper_feed_tick_once

        try:
            return run_paper_feed_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"paper feed tick failed: {exc}"}

    @registry.register(
        "__group_ritual_tick__",
        description="Run one group ritual tick (audit gap R2-29): "
        "every themed community group with a ritual due within a "
        "minute gets it rendered and posted into the group feed as "
        "Devon, then marked delivered. Scheduler-internal; runs "
        "every ~5m. Routine and silent — a run with nothing due "
        "posts nothing.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _group_ritual_tick_tool() -> dict[str, Any]:
        from ..community.digest import run_group_ritual_tick_once

        try:
            return run_group_ritual_tick_once()
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"group ritual tick failed: {exc}"}

    @registry.register(
        "__watch_loop_tick__",
        description="Run one resident watch-loop tick (audit gap R2-22): "
        "re-scan the ops surfaces (stuck missions with one self-heal "
        "attempt, chronic routing roles, unrecovered pivots), redeliver "
        "the durable pending-alert queue the moment a platform is "
        "reachable again, and re-mine the reasoning journal for new "
        "prevention skills. Scheduler-internal; runs every ~5m. "
        "Routine and silent — an all-quiet tick pages nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _watch_loop_tick_tool() -> dict[str, Any]:
        from ..agents.watcher import WatchLoop

        try:
            return {"ok": True, **WatchLoop(_context).tick()}
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"watch loop tick failed: {exc}"}

    @registry.register(
        "__health_drift_tick__",
        description="Run one health drift tick (audit gap R2-28): "
        "check the owner's 3-day sleep/mood/HRV/activity trends "
        "against their baselines and surface a drifted signal — an "
        "\"act\" report pings the owner once a day at most, a "
        "\"watch\" report is returned for the morning briefing and "
        "never pinged. Scheduler-internal; runs daily at 07:00. "
        "Routine and silent — a stable timeline pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _health_drift_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..health.drift import run_health_drift_tick_once

        try:
            return run_health_drift_tick_once(
                context=_context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"health drift tick failed: {exc}"}

    @registry.register(
        "__workflow_tick__",
        description="Run one workflow tick: start every queued "
        "workflow script on a background thread, reap finished "
        "threads, deliver finished summaries into the origin chat, "
        "and mark runs whose thread died with the process "
        "interrupted (record + journal survive; the owner re-runs "
        "the saved script). Scheduler-internal; runs every ~5m. "
        "Routine and silent — nothing queued means nothing happens.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _workflow_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..agents.workflow.runner import run_workflow_tick_once

        try:
            return run_workflow_tick_once(
                _context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"workflow tick failed: {exc}"}

    @registry.register(
        "__swarm_board_tick__",
        description="Run one swarm-board tick: wake stalled swarm boards — a board quiet past 10 minutes with unfinished work gets one @monitor wake-up naming the unfinished workers, and the owner is told once per stall episode. Scheduler-internal; runs every ~5m. Routine and silent — a live board pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _swarm_board_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..agents.swarm_board import run_swarm_board_tick_once

        try:
            return run_swarm_board_tick_once(
                _context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"swarm board tick failed: {exc}"}

    @registry.register(
        "__continual_update_tick__",
        description="Run one continual-update tick: harvest fresh owner-correction pairs from the representation ledger into the durable replay buffer and, when due (every ~24h), run one confined adapter-only update against the frozen eval harness. An accepted candidate is saved as the live preference adapter and the owner is told once; a rejection leaves the previous adapter in place. Scheduler-internal; runs every ~24h. Routine and silent — a not-due or deferred tick pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _continual_update_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..training.continual_driver import run_continual_update_once

        try:
            return run_continual_update_once(
                _context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"continual update tick failed: {exc}"}

    @registry.register(
        "__distill_tick__",
        description="Run one distillation tick: harvest kept cloud turns from the conversation log into the durable distillation store and, when enough pairs exist (32+), train the local adapter and run it through the TrainingRegistry promotion gate. A promoted adapter notifies the owner once; a failed gate means the candidate never lands. Scheduler-internal; runs every ~24h. Routine and silent — a deferred or not-due tick pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _distill_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..training.distill_driver import run_distill_once

        try:
            return run_distill_once(
                _context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"distill tick failed: {exc}"}

    @registry.register(
        "__visual_idle_tick__",
        description="Run one visual idle tick: when the owner has gone idle after acting (danpeg's acted-since-last-check gate — a pause can never fire twice, and an AFK owner never fires at all), and a workflow or plan is still open, publish exactly one note naming it. Scheduler-internal; runs every ~5m. Routine and silent — a tick with nothing open pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _visual_idle_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..visuals.idle import run_visual_idle_tick_once

        try:
            settings = getattr(_context, "settings", None)
            home = getattr(settings, "home", None) if settings else None
            return run_visual_idle_tick_once(
                home, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"visual idle tick failed: {exc}"}

    @registry.register(
        "__render_tick__",
        description="Run one render tick: every non-terminal render-tagged "
        "mission advances exactly one stage (proxy, then full) through "
        "the real render engine; a stage failure is recorded on the "
        "mission and retried next tick, three in a row fails it, and a "
        "finished render notifies the owner once with the output path. "
        "Scheduler-internal; runs every ~5m. Routine and silent — a "
        "tick with nothing due pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _render_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..missions.render_driver import run_render_tick_once

        try:
            return run_render_tick_once(
                _context, notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"render tick failed: {exc}"}

    @registry.register(
        "__offline_sync_tick__",
        description="Run one offline sync tick: drain the durable offline queue through the sync engine (FIFO, conflicts flagged, failed actions retried), pump the gateway outbox's queued replies, and publish any new sync conflict once. Scheduler-internal; runs every ~5m. Routine and silent — a tick with nothing queued pings nobody.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _offline_sync_tick_tool() -> dict[str, Any]:
        from ..agents.notifier import Notifier
        from ..agents.offline_host import (
            get_offline_host,
            run_offline_sync_tick_once,
        )

        try:
            extras = getattr(_context, "extras", None) or {}
            gateway = extras.get("gateway") if isinstance(extras, dict) else None
            return run_offline_sync_tick_once(
                _context, host=get_offline_host(), gateway=gateway,
                notifier=Notifier(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"offline sync tick failed: {exc}"}

    @registry.register(
        "__show_schedule_tick__",
        description="Run one show-schedule tick: compose a DRAFT "
        "episode for every show whose schedule is due, from the "
        "latest stored morning briefing (the pulse) and fresh "
        "exploration claims (the research). Scheduler-internal; "
        "runs every ~15m. Routine and silent — it never publishes "
        "and never posts anywhere; publishing stays the owner's "
        "tap on /show publish.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _show_schedule_tick_tool() -> dict[str, Any]:
        from ..shows.scheduling import run_show_schedule_tick_once

        try:
            return run_show_schedule_tick_once(
                context=_context,
                pulse=lambda: _show_pulse_sections(_context),
                research=lambda: _show_research_sections(_context))
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"show schedule tick failed: {exc}"}

    @registry.register(
        "__unified_scheduler_tick__",
        description="Run one unified-scheduler pass: every due mission "
        "and organ tick on the boot-built autonomy stack, under the "
        "rhythm + budget gates, each firing explained into the why "
        "store. Scheduler-internal; runs every ~5m. Routine and "
        "silent — the driver the unified scheduler never had: before "
        "this wiring only `nm why run` ever called run_due(), so a "
        "scheduled mission fired only if someone happened to run it.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _unified_scheduler_tick_tool() -> dict[str, Any]:
        from ..autonomy.unified import run_unified_scheduler_tick_once

        try:
            # The ONE stack boot built (context.extras): organ ticks
            # and mission handlers registered on it survive across
            # ticks. When boot never ran (a bare tool call, a test),
            # build a fresh stack from the same home — the mission
            # queue and the why log are files, so a fresh stack sees
            # the same durable state.
            stack = (getattr(_context, "extras", None) or {}).get(
                "autonomy_stack")
            if stack is None:
                home = getattr(
                    getattr(_context, "settings", None), "home", None)
                if not home:
                    return {"ok": False,
                            "error": "unified scheduler tick: no "
                            "autonomy stack on this runtime and no "
                            "nomorals home to build one from"}
                from ..autonomy.wiring import build_stack

                stack = build_stack(home)
            return run_unified_scheduler_tick_once(stack.scheduler)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"unified scheduler tick failed: {exc}"}

    @registry.register(
        "__delegation_manager_tick__",
        description="Run one delegation-manager tick: recover "
        "delegated runs orphaned by a restart, resume what the "
        "restart interrupted, finalize finished workers, and "
        "collect completed runs (delivering their results into the "
        "owner's chat via the notifier). Scheduler-internal; runs "
        "every ~5m. Never raises — a bad tick is recorded, never "
        "fatal to the job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _delegation_manager_tick_tool() -> dict[str, Any]:
        from ..agents.delegation_manager import (
            run_delegation_manager_tick_once,
        )

        try:
            return run_delegation_manager_tick_once(_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"delegation manager tick failed: {exc}"}

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
