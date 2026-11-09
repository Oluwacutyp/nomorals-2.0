"""Startup wiring for the agency layer (audit §9 step 1).

``boot_agency(db, ...)`` is the ONE entrypoint the partner runtime
calls on boot (best-effort, never breaks boot). It runs the
supervision reconcile and returns the store + supervisor so the
runtime can keep them. Idempotent: safe to call on every boot.

The worker SHOWS UP: when a scheduler is given, boot registers the
durable ``agency-shift`` job (``__agency_shift__`` tool, every 30
min) so shifts run on schedule, not only when poked.
"""
from __future__ import annotations

from typing import Any

from .adapters import AgencyAuthority, build_adapters
from .expansion import ExpansionSystems
from .initiatives import InitiativeStore
from .supervision import BootSupervisor
from .systems import ActiveSystems

__all__ = [
    "boot_agency", "ensure_agency_shift_job", "run_scheduled_expansion",
    "compose_agency", "resolve_gateway", "search_fn_from_context",
    "durable_permission_gradient",
    "AGENCY_SHIFT_JOB", "AGENCY_SHIFT_TOOL", "AGENCY_SHIFT_SPEC",
]

#: Durable shift job: name, tool payload, default cadence.
AGENCY_SHIFT_JOB = "agency-shift"
AGENCY_SHIFT_TOOL = "__agency_shift__"
AGENCY_SHIFT_SPEC = "*/30 * * * *"

#: Generous wall-clock cap for the shift job. The scheduler's default
#: ``wall_seconds`` (300s) is a blunt timeout with thread abandonment:
#: a shift legitimately running longer was marked failed, paged the
#: owner falsely, and lost its result. 1800s is the shift's own
#: interval — a job that cannot finish inside its own slot is genuinely
#: wedged. (Heartbeat decision: generous last-resort hang-guards only.)
SHIFT_RUN_TIMEOUT_S = 1800.0


def _shift_spec_matches(stored: Any, spec: str) -> bool:
    """True when a stored job spec is the requested one.

    ``Scheduler.list_jobs()`` formats cron specs with a ``"cron "``
    prefix (``"cron */30 * * * *"``) while callers pass the bare
    expression — both forms must compare, or the idempotency check can
    never succeed and every boot re-registers the job (ID churn,
    orphaned run history, lost retry state).
    """
    s = str(stored or "").strip()
    want = (spec or "").strip()
    if s == want:
        return True
    if s.startswith("cron "):
        s = s[5:].strip()
    return s == want


def _job_run_timeout(job: Any) -> float:
    raw = job.get("run_timeout_s", 0.0) if isinstance(job, dict) \
        else getattr(job, "run_timeout_s", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def ensure_agency_shift_job(scheduler: Any, *,
                            spec: str = AGENCY_SHIFT_SPEC) -> Any:
    """Register the agency-shift job. Idempotent by job name.

    A stale job (different interval OR different run cap) is replaced.
    Best-effort: never raises (a missing scheduler just means no
    scheduled shifts — the runtime can still call ``work_shift``
    directly).
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = job.get("name", "") if isinstance(job, dict) \
                else getattr(job, "name", "")
            if name == AGENCY_SHIFT_JOB:
                job_spec = job.get("spec", "") if isinstance(job, dict) \
                    else getattr(job, "spec", "")
                if (_shift_spec_matches(job_spec, spec)
                        and _job_run_timeout(job) == SHIFT_RUN_TIMEOUT_S):
                    return job
                job_id = job.get("id", "") if isinstance(job, dict) \
                    else getattr(job, "id", "")
                try:
                    scheduler.remove(job_id)
                except Exception:  # noqa: BLE001 - stale job removal
                    pass
                break
        return scheduler.add(AGENCY_SHIFT_JOB, spec, "tool",
                             {"tool": AGENCY_SHIFT_TOOL},
                             run_timeout_s=SHIFT_RUN_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def resolve_gateway(context: Any) -> Any:
    """The ONE gateway resolution for agency composition (F09).

    The partner runtime stores the live gateway in
    ``context.extras['gateway']`` (runtime.py: ``context.extras[
    "gateway"] = self.gateway``); some contexts carry it as a plain
    attribute. The scheduled shift used to read ONLY the attribute —
    which the real AgentContext never has — so every scheduled shift
    ran gateway-less (no DM sender, social adapter blind) while the
    gateway sat in extras. Resolution order: extras first (the
    runtime's documented binding), then the attribute, else None.
    """
    if context is None:
        return None
    try:
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            gateway = extras.get("gateway")
            if gateway is not None:
                return gateway
    except Exception:  # noqa: BLE001 - a broken extras must not hide the attr
        pass
    return getattr(context, "gateway", None)


def search_fn_from_context(context: Any) -> Any:
    """The ONE live-search seam for agency composition (F09).

    Identical construction to the interactive money paths
    (``agents/opportunities.handle_money_command`` and
    ``WeeklyOpportunityHunt.from_context``): the context's
    ``web_search`` tool, adapted to the hunter's
    ``search_fn(query, max_results) -> [{url, title, snippet}]``
    shape. None when the context carries no tool registry — the
    hunter then scans its curated pool only, and the composition
    readiness report says so.
    """
    tools = getattr(context, "tools", None) if context is not None else None
    if tools is None:
        return None

    def search_fn(q: str, n: int = 6) -> list:
        outcome = tools.call("web_search", query=q, max_results=n)
        if not outcome.ok:
            err = outcome.error.message if outcome.error else "unknown"
            raise RuntimeError(f"search failed: {err}")
        return list(outcome.unwrap().get("results") or [])

    return search_fn


def durable_permission_gradient(db: Any) -> Any:
    """Permission gradient from the DURABLE policy allow-rules.

    Loads ``Policy`` rules persisted in ``db`` (``Policy.load_rules``)
    so scheduled and interactive agency work run under the SAME
    authority — not a weaker default-deny shadow. Missing/unreadable
    rules = no persisted rules = default-deny stands (a failure here
    narrows authority, never widens it).
    """
    from ..core.policy import PermissionGradient, Policy

    policy = Policy()
    try:
        policy.load_rules(db)
    except Exception:  # noqa: BLE001 - unreadable rules = default-deny
        pass
    return PermissionGradient(policy)


def compose_agency(context: Any, *, db: Any = None) -> dict[str, Any]:
    """THE agency composition root (F09) — one contract, both paths.

    The scheduled shift (``tools/autonomy.py``) and the interactive
    runtime boot (``boot_agency(..., context=...)``) both build their
    agency systems HERE, so scheduled autonomy runs with the same
    real gateway, live search, and organs as interactive work — not
    a degraded twin assembled from memory. Every organ construction
    is best-effort: a failed organ degrades to None (an organ-less
    adapter / an honest blocker downstream), never a crash — and the
    returned ``readiness`` map states, per advertised system, whether
    it is ready or its concrete unavailable/delegated state. Nothing
    here counts "adapter constructed" as "feature ran": readiness is
    a wiring statement; the shift's receipts are the run statement.

    Returns a dict with ``db``, ``adapters``, ``authority``,
    ``gateway``, ``search_fn``, ``research_organ``, ``wisdom_organ``,
    ``impossible``, and ``readiness``.
    """
    # Call-time import: build_adapters is resolved from the adapters
    # module at call time (the shift tests' patch point), and the
    # organ imports below stay lazy so this L5 composition module
    # never drags organs into importers that only need the boot API.
    import functools
    from pathlib import Path
    from types import SimpleNamespace

    from .adapters import AgencyAuthority, build_adapters

    def _try(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - organ unavailable = None
            return None

    settings = getattr(context, "settings", None)
    ws = getattr(settings, "workspace_dir", None) or "."
    if db is None:
        db = getattr(context, "db", None)
    if db is None:
        from ..autonomy.idle import workspace_db
        db = workspace_db(ws)
    gateway = resolve_gateway(context)
    search_fn = search_fn_from_context(context)

    def _news_agent():
        from ..agents.news import NewsAgent
        return NewsAgent(context)

    def _social_manager():
        from ..social.manager import SocialManager
        return SocialManager(context) if context is not None else None

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
        return OpportunityHunter(settings=settings, search_fn=search_fn)

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

    def _dm_sender(gw):
        # The real DM-send channel: the chat gateway's send path.
        # None when no gateway is wired — sends then block honestly.
        if gw is None:
            return None

        def _send(platform, contact, body):
            try:
                res = gw.send(platform, contact, body)
                return bool(getattr(res, "ok", False))
            except Exception:  # noqa: BLE001 - send failure = not sent
                return False

        return _send

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
        gradient=_try(lambda: durable_permission_gradient(db)),
        grant=None, mandate_check=_try(_mandate_check))

    # The initiative organs (research / wisdom / impossible) join the
    # SAME composition: the ActiveSystems built from this dict works
    # their per-trigger handlers against the real organs, so an
    # initiative's lifecycle ends in the organ's own receipt — the
    # identities are joined here instead of a shadow organ-less path
    # (their independent idle ticks in autonomy/coordinator remain,
    # working the same stores).
    organ_context = context
    if organ_context is None or getattr(organ_context, "db", None) is None:
        organ_context = SimpleNamespace(
            db=db, settings=settings,
            tools=getattr(context, "tools", None),
            extras=getattr(context, "extras", None) or {})

    def _research():
        from ..research.autonomy import ResearchOrgan
        return ResearchOrgan(organ_context)

    def _wisdom():
        from ..wisdom.autonomy import WisdomOrgan
        return WisdomOrgan(organ_context)

    def _impossible():
        from ..impossible.core import ImpossibleMadePossible
        base = Path(str(ws)) / "data"
        return ImpossibleMadePossible(
            ledger_path=base / "impossible_ledger.json",
            queue_path=base / "impossible_queue.json",
            spec_dir=base / "impossible_specs")

    research_organ = _try(_research)
    wisdom_organ = _try(_wisdom)
    impossible = _try(_impossible)

    gigs_adapter = adapters.get("gigs")
    readiness = {
        "gateway": ("ready" if gateway is not None else
                    "unavailable: no live gateway on the context "
                    "(extras['gateway'] unset) — DM sends block "
                    "honestly, drafts are held"),
        "search": ("ready: live web_search tool backs the opportunity "
                   "hunter" if search_fn is not None else
                   "unavailable: the context carries no tool registry "
                   "— the opportunity hunter scans its curated pool "
                   "only, no fresh external discovery"),
        "gigs": ("ready" if gigs_adapter is not None
                 and getattr(gigs_adapter, "hunter", None) is not None
                 and getattr(gigs_adapter, "applier", None) is not None
                 else "unavailable: gig hunter/applier construction "
                      "failed — gig triggers block honestly"),
        "research": ("ready" if research_organ is not None else
                     "unavailable: research organ construction failed "
                     "— its independent coordinator tick still runs"),
        "wisdom": ("ready" if wisdom_organ is not None else
                   "unavailable: wisdom organ construction failed — "
                   "its independent coordinator tick still runs"),
        "impossible": ("ready" if impossible is not None else
                       "unavailable: impossible organ construction "
                       "failed"),
        "deals": ("delegated: no synchronous deal hunter or purchase "
                  "executor exists in production (the Naija deal "
                  "hunter is async; the deals adapter is synchronous) "
                  "— the adapter observes nothing, and purchases stay "
                  "blocked at its gates, never faked"),
        "planner": ("delegated: deliberation is the planner's "
                    "executor on this path; no durable plan-graph "
                    "queue exists — injected graphs execute when "
                    "supplied"),
        "expansion": ("delegated: expansion domain organs run on "
                      "their own schedules; source-health "
                      "re-verification runs in the shift via "
                      "run_scheduled_expansion"),
    }
    return {"db": db, "adapters": adapters, "authority": authority,
            "gateway": gateway, "search_fn": search_fn,
            "research_organ": research_organ,
            "wisdom_organ": wisdom_organ, "impossible": impossible,
            "readiness": readiness}


def run_scheduled_expansion(db: Any, *, organs: dict[str, Any] | None = None,
                            authority: Any | None = None,
                            explicit_go_aheads: Any | None = None,
                            registry_collect: Any | None = None,
                            registry_fetcher: Any = None,
                            budget_s: float = 600.0,
                            now: float | None = None) -> dict[str, Any]:
    """Run one scheduled expansion cycle (the SourceHealth entrypoint).

    This is the scheduler/daemon entrypoint for expansion: the
    ``agency-shift`` job calls it after the core ``work_shift``, so
    SourceHealth actually re-verifies sources on schedule — not just
    as an existing-but-never-called tick. Defaults wire the real
    registry collector (``sources_registry.collect``) and the real
    ``SourceHealth``; callers may inject fakes for tests. Expansion
    organs run under standing authority (gated actions go through
    the trial-pattern gate).
    """
    from .sourcehealth import SourceHealth

    if registry_collect is None:
        from .sources_registry import collect as _collect

        def registry_collect() -> list:
            specs, _ = _collect()
            return specs
    systems = ExpansionSystems(
        db=db, organs=organs, source_health=SourceHealth(db),
        registry_collect=registry_collect, registry_fetcher=registry_fetcher,
        authority=authority, explicit_go_aheads=explicit_go_aheads)
    systems.seed_initiatives(now=now)
    return systems.run_cycle(budget_s=budget_s, now=now)


def boot_agency(db: Any, *, now: float | None = None,
                scheduler_facade: Any | None = None,
                watches: list[dict[str, Any]] | None = None,
                research_organ: Any | None = None,
                wisdom_organ: Any | None = None,
                impossible: Any | None = None,
                adapters: dict[str, Any] | None = None,
                authority: Any | None = None,
                auto_wire: bool = True,
                context: Any | None = None,
                account_manager: Any | None = None,
                places: list[str] | None = None,
                inbox: Any | None = None,
                scheduler: Any | None = None,
                shift_spec: str = AGENCY_SHIFT_SPEC,
                ) -> dict[str, Any]:
    store = InitiativeStore(db)
    supervisor = BootSupervisor(store, scheduler_facade=scheduler_facade)
    report = supervisor.reconcile(now=now, watches=watches)
    # ALL AI systems actively working: seed one standing initiative per
    # system (idempotent via dedupe_key) so every system enters the
    # rotation with a live lifecycle on every boot.
    if adapters is None and auto_wire:
        if context is not None:
            # The interactive boot composes through the SAME root as
            # the scheduled shift (F09): one composition contract for
            # both paths. Explicitly-passed organs/authority win;
            # composition fills only what the caller did not supply.
            # Best-effort: a composition failure falls back to the
            # organ-less assembly, never a broken boot.
            try:
                comp = compose_agency(context, db=db)
            except Exception:  # noqa: BLE001 - boot must not break
                comp = None
            if comp is not None:
                adapters = comp["adapters"]
                if research_organ is None:
                    research_organ = comp["research_organ"]
                if wisdom_organ is None:
                    wisdom_organ = comp["wisdom_organ"]
                if impossible is None:
                    impossible = comp["impossible"]
                if authority is None:
                    authority = comp["authority"]
        if adapters is None:
            # Assembly only: no organs are constructed here (layering —
            # organs are injected by the caller). Real organs arrive
            # via ``adapters=`` or the composition root above.
            adapters = build_adapters(places=places, inbox=inbox)
    if authority is None:
        authority = AgencyAuthority()
    systems = ActiveSystems(db=db, store=store,
                            research_organ=research_organ,
                            wisdom_organ=wisdom_organ,
                            impossible=impossible,
                            adapters=adapters,
                            authority=authority)
    seeded = systems.seed_initiatives(now=now)
    # The worker SHOWS UP: register the durable shift job so shifts run
    # on schedule, not only when poked. Best-effort, never breaks boot.
    shift_job = None
    if scheduler is not None:
        shift_job = ensure_agency_shift_job(scheduler, spec=shift_spec)
    return {"store": store, "supervisor": supervisor, "boot": report,
            "systems": systems, "seeded_initiatives": len(seeded),
            "shift_job": shift_job}
