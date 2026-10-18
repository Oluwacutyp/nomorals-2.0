"""Real organ adapters for the agency domain loops.

Each adapter is a thin translation layer between an agency loop and
the REAL organ that does the work — never a re-implementation, never
a fabrication. Organs are INJECTED by the caller (a higher layer):
the agency layer is L0 (ARCHITECTURE.md §1) and must not import any
organ (``core`` / ``agents`` / ``social`` / ``planning`` /
``integrations`` / ``finance``). An adapter without its organ returns
empty observations and refuses actions — it never fakes organ output.

Standing authority is NOT re-implemented here: :class:`AgencyAuthority`
is a mapping onto the EXISTING systems — ``core/policy.py``
(PermissionGradient + Policy, injected) for action gates,
``finance/mandate.py`` (PaymentMandate, via an injected
``mandate_check`` callable) for spending, and the trial pattern
(``agents/trial/flow.py``: explicit confirmation for THAT signup/gig)
for explicit go-aheads.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .domains import (
    advance_lead, classify_weather_item, cluster_news, evaluate_deal,
    score_gig,
)

__all__ = [
    "AgencyAuthority", "DealsAdapter", "WeatherAdapter", "NewsAdapter",
    "SocialAdapter", "LeadsAdapter", "GigsAdapter", "PlannerAdapter",
    "LEAD_STORE_MAP", "build_adapters",
]

#: Agency lifecycle stage → LeadStore coarse status. Stages that share
#: a store status are distinguished by the agency initiative (evidence
#: + next action), never by pretending the store tracks them.
LEAD_STORE_MAP = {
    "discovered": "new", "qualified": "new",
    "contacted": "dmed", "replied": "dmed", "scoped": "dmed",
    "proposal": "dmed", "won": "converted", "delivery": "converted",
    "paid": "converted", "follow_up": "followed_up", "lost": "dead",
}


@dataclass
class _DeniedDecision:
    """Fail-closed decision when NO authority gradient is injected.

    Only ever denies — it can never allow. Real decisions come from
    the injected ``core.policy`` gradient; this shape exists so the
    agency layer never imports ``core`` (layering: agency is L0)
    and never fabricates an approval.
    """
    allowed: bool = False
    reason: str = "no authority gradient injected — fail closed"
    capability: str = ""
    actor: str = ""
    needs_confirmation: bool = False
    needs_biometric: bool = False
    needs_proposal: bool = False
    proposal_id: str = ""
    gradient: str = ""
    explicit_override: bool = False
    audit_id: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class AgencyAuthority:
    """Agency's view of the EXISTING standing-authority systems.

    * Action gates → ``core/policy.py`` PermissionGradient + Policy,
      INJECTED as ``gradient`` (the agency layer never imports
      ``core`` — layering: agency is L0, organs are injected).
    * Spending → ``finance/mandate.py`` PaymentMandate, via an
      injected ``mandate_check`` callable
      ``(principal, scope, amount_kobo, *, ledger) -> check`` with
      ``.ok`` / ``.reason`` (e.g. ``functools.partial(check_mandate,
      store)`` from the caller).
    * Explicit per-gig go-ahead → the trial pattern
      (``agents/trial/flow.py``): explicit confirmation for THAT gig.

    Defaults: no injected gradient + no mandate checker = nothing
    outbound executes autonomously. Grants / allow rules / mandates
    come from the owner through the existing policy/mandate flows —
    this class never mints authority by itself.
    """

    #: Capability strings. ``core.policy.Capability`` is strings, not
    #: an enum, precisely so tools can extend it — ``gig.submit`` is
    #: such an extension (no gig capability existed yet).
    CAP_POST = "social.post"
    CAP_DM = "social.dm"
    CAP_SUBMIT = "gig.submit"

    gradient: Any = None
    grant: Any = None
    actor: str = "agency"
    explicit_submit_ids: frozenset[str] = frozenset()
    mandate_check: Any = None
    ledger: Any = None
    principal: str = "owner"
    #: Mandate scope for deal spending. ``PaymentMandate.covers`` lets
    #: an ``"all"`` mandate cover this too.
    spend_scope: str = "purchase"

    def __post_init__(self) -> None:
        self.explicit_submit_ids = frozenset(self.explicit_submit_ids or ())

    # -- action gates (core/policy.py, injected) --
    def check(self, capability: str, *, explicit_override: bool = False,
              draft: str = "", context: dict[str, Any] | None = None) -> Any:
        """One policy+gradient decision. Never raises (fail closed)."""
        if self.gradient is None:
            return _DeniedDecision(
                reason="no authority gradient injected — fail closed "
                       "(defaults deny)",
                capability=capability, actor=self.actor)
        return self.gradient.check(
            capability, actor=self.actor, grant=self.grant,
            autonomous=not explicit_override,
            explicit_override=explicit_override,
            draft=draft, context=context)

    def may_post(self, *, draft: str = "") -> Any:
        """Autonomous post decision (gradient: propose unless allowed)."""
        return self.check(self.CAP_POST,
                          draft=draft or "autonomous social post")

    def may_dm(self, *, draft: str = "") -> Any:
        """Autonomous DM decision (``social.dm`` is confirmable)."""
        return self.check(self.CAP_DM, draft=draft or "autonomous DM")

    def may_submit(self, gig_id: str, *, explicit: bool = False,
                   draft: str = "") -> Any:
        """Gig-submit decision: trial-pattern explicit go-ahead + policy.

        * ``explicit=True`` + gig in ``explicit_submit_ids`` → the
          owner's explicit instruction for THAT gig: gradient bypassed,
          but deny rules, grants, and biometric still apply.
        * Otherwise the gradient's propose level records a proposal —
          never a silent submit.
        """
        if explicit and gig_id in self.explicit_submit_ids:
            return self.check(self.CAP_SUBMIT, explicit_override=True,
                              context={"gig_id": gig_id})
        if explicit:
            return _DeniedDecision(
                reason=(f"no explicit go-ahead for gig {gig_id!r} — "
                        "submit needs the owner's explicit instruction "
                        "for THAT gig (trial pattern)"),
                capability=self.CAP_SUBMIT, actor=self.actor)
        return self.check(self.CAP_SUBMIT,
                          draft=draft or f"submit gig {gig_id}",
                          context={"gig_id": gig_id})

    # -- spending gate (finance/mandate.py, injected) --
    def spend_decision(self, amount: float, *,
                       scope: str | None = None) -> Any:
        """Pure mandate check for a planned spend (Naira → kobo).

        Returns None when no mandate checker is injected (no mandate
        = no spending). Otherwise the mandate system's own check
        with its reason.
        """
        if self.mandate_check is None:
            return None
        amount_kobo = int(round(float(amount) * 100))
        return self.mandate_check(self.principal, scope or self.spend_scope,
                                  amount_kobo, ledger=self.ledger)

    def can_spend(self, amount: float, *, scope: str | None = None) -> bool:
        decision = self.spend_decision(amount, scope=scope)
        return bool(decision is not None and decision.ok)

    def spend_reason(self, amount: float, *,
                     scope: str | None = None) -> str:
        decision = self.spend_decision(amount, scope=scope)
        if decision is None:
            return ("no payment mandate — the owner must issue one with "
                    "/mandate issue before money can move")
        return decision.reason or "mandate covers this spend"


# ── Deals ──────────────────────────────────────────────────────────

class DealsAdapter:
    """Universal deals engine (need → exact product → landed vs
    historical → verdict). ``integrations/naija_deals.py`` is ONE
    regional adapter that can sit behind ``hunter`` — never the
    organizing axis. Never purchases: ``check_need`` verifies and
    reports readiness / blockers; buying needs a mandate + executor
    the owner approves."""

    def __init__(self, hunter: Any | None = None,
                 price_check: Any | None = None) -> None:
        self.hunter = hunter
        self.price_check = price_check

    @staticmethod
    def _as_dicts(items: Any) -> list[dict[str, Any]]:
        out = []
        for item in (items or []):
            if isinstance(item, dict):
                out.append(dict(item))
            else:
                to_dict = getattr(item, "to_dict", None)
                out.append(dict(to_dict()) if callable(to_dict)
                           else {"value": str(item)})
        return out

    def check_alerts(self) -> list[dict[str, Any]]:
        """Triggered price alerts from the hunter (tracked needs)."""
        if self.hunter is None:
            return []
        try:
            return self._as_dicts(self.hunter.check_alerts())
        except Exception:  # noqa: BLE001 - hunter failure = no alerts
            return []

    def search(self, query: str) -> list[dict[str, Any]]:
        if self.hunter is None:
            return []
        try:
            return self._as_dicts(self.hunter.search(query))
        except Exception:  # noqa: BLE001
            return []

    def price_detail(self, product: str) -> dict[str, Any]:
        if self.hunter is None:
            return {}
        try:
            out = self.hunter.price_detail(product)
            return dict(out) if isinstance(out, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def price_history(self, product: str) -> dict[str, Any]:
        if self.hunter is None:
            return {}
        try:
            out = self.hunter.price_history(product)
            return dict(out) if isinstance(out, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def check_need(self, need: dict[str, Any]) -> dict[str, Any]:
        """Actually CHECK a need: price, landed, historical, verdict.

        Necessity gate stays: 60% off unnecessary ≠ savings;
        landed < historical low for the EXACT variant.
        """
        product = str(need.get("product", need.get("query", "")))
        observed: dict[str, Any] = {}
        vendor = ""
        url = ""
        if self.hunter is not None:
            try:
                detail = self.price_detail(product)
                observed.update(detail)
                vendor = str(detail.get("vendor", "") or "")
                url = str(detail.get("url", "") or "")
            except Exception:  # noqa: BLE001
                pass
            try:
                hist = self.price_history(product)
                if hist:
                    observed["historical_low"] = hist.get(
                        "lowest_price", need.get("historical_low"))
            except Exception:  # noqa: BLE001
                pass
        elif self.price_check is not None:
            try:
                observed.update(dict(self.price_check(need) or {}))
            except Exception:  # noqa: BLE001
                pass
        price = float(observed.get("price", observed.get("current_price", 0) or 0))
        landed = price + float(need.get("shipping", 0) or 0) \
            + float(need.get("fees", 0) or 0)
        if "landed_price" in observed:
            landed = float(observed["landed_price"])
        historical_low = float(observed.get(
            "historical_low", need.get("historical_low", float("inf"))))
        discount = float(observed.get("discount_percent",
                                      observed.get("discount_pct", 0) or 0))
        evaluation = evaluate_deal(
            needed=bool(need.get("needed", True)), discount_pct=discount,
            price=price, landed_price=landed, historical_low=historical_low,
            fresh=bool(observed.get("fresh", True)),
            variant_matches=bool(observed.get("variant_matches", True)))
        return {"evaluation": evaluation,
                "observed": {"price": price, "landed_price": landed,
                             "historical_low": historical_low,
                             "fresh": observed.get("fresh", True),
                             "in_stock": observed.get("in_stock", True),
                             "vendor": vendor, "url": url}}


# ── Weather ────────────────────────────────────────────────────────

class WeatherAdapter:
    """Real ``agents/weather.py`` forecasts + official warnings."""

    def __init__(self, weather: Any | None = None,
                 places: list[str] | None = None) -> None:
        self._weather = weather
        self._places = list(places or [])

    def _organ(self) -> Any:
        # The organ is INJECTED by the caller (higher layer) — the
        # agency layer never imports it (layering). No organ = no
        # observations, never fabricated weather.
        return self._weather

    def places(self) -> list[str]:
        return list(self._places)

    def snapshot(self, place: str) -> dict[str, Any]:
        """One place: forecast days + classified alerts (no network lies).

        Alerts from the organ are classified: NWS items are official
        warnings (US-only, enforced); anything else stays a forecast.
        """
        try:
            organ = self._organ()
            fc = organ.forecast(place) or {}
            now = organ.now(place) or {}
        except Exception:  # noqa: BLE001
            return {"place": place, "days": [], "alerts": []}
        alerts: list[dict[str, Any]] = []
        for line in (now.get("alerts") or []):
            try:
                alerts.append(classify_weather_item(
                    kind="nws_alert", region="US", text=str(line)))
            except ValueError:
                alerts.append(classify_weather_item(
                    kind="forecast_heuristic", region="US", text=str(line)))
        return {"place": place,
                "days": list(fc.get("days") or []),
                "alerts": alerts,
                "ok": bool(fc.get("ok"))}

    def forecast(self, place: str, days: int = 3) -> dict[str, Any]:
        try:
            return dict(self._organ().forecast(place, days=days) or {})
        except Exception:  # noqa: BLE001
            return {"ok": False, "place": place}


# ── News ───────────────────────────────────────────────────────────

class NewsAdapter:
    """Real ``agents/news.py`` feed acquisition + digests."""

    def __init__(self, agent: Any | None = None,
                 agent_factory: Any | None = None) -> None:
        self._agent = agent
        self._factory = agent_factory

    def _organ(self) -> Any:
        if self._agent is None and self._factory is not None:
            self._agent = self._factory()
        return self._agent

    def recent(self, limit: int = 15) -> list[dict[str, Any]]:
        organ = self._organ()
        if organ is None:
            return []
        try:
            items = organ.recent(limit=limit) or []
        except Exception:  # noqa: BLE001
            return []
        return [dict(i) if isinstance(i, dict) else {"title": str(i)}
                for i in items]

    def fetch_digest(self) -> dict[str, Any]:
        """Full digest run through the organ (feed acquisition +
        summaries), not just the cached recent list."""
        organ = self._organ()
        if organ is None:
            return {}
        try:
            out = organ.run()
            return dict(out) if isinstance(out, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def clusters(self, limit: int = 15) -> list[dict[str, Any]]:
        """Clustered recent stories: syndication is ONE story."""
        return cluster_news(self.recent(limit=limit))


# ── Social ─────────────────────────────────────────────────────────

class SocialAdapter:
    """Real social executors (posting, community, DM handling).

    ``manager`` is the real ``social/manager.py`` SocialManager
    (``due()`` / ``run_due()``). ``inbox`` is an injected callable
    returning the unanswered / promised / milestone backlog
    (``{"unanswered", "promised_replies", "milestones",
    "stop_signal", "cadence_ok"}``) — no default DM source is
    fabricated. Drafts are built, never sent, without authority.
    """

    def __init__(self, manager: Any | None = None,
                 inbox: Any | None = None) -> None:
        self.manager = manager
        self.inbox = inbox

    def backlog(self) -> dict[str, Any]:
        if self.inbox is None:
            return {"unanswered": 0, "promised_replies": 0,
                    "milestones": 0, "stop_signal": False,
                    "cadence_ok": True}
        try:
            out = self.inbox()
            return dict(out) if isinstance(out, dict) else {}
        except Exception:  # noqa: BLE001
            return {"unanswered": 0, "promised_replies": 0,
                    "milestones": 0, "stop_signal": False,
                    "cadence_ok": True}

    def due_posts(self) -> list[dict[str, Any]]:
        if self.manager is None:
            return []
        try:
            due = self.manager.due() or []
        except Exception:  # noqa: BLE001
            return []
        out = []
        for item in due:
            if isinstance(item, dict):
                out.append(dict(item))
            else:
                to_dict = getattr(item, "to_dict", None)
                out.append(dict(to_dict()) if callable(to_dict)
                           else {"value": str(item)})
        return out

    def publish_due(self) -> list[dict[str, Any]]:
        """Publish due posts through the organ. Authority-gated by
        the caller (``systems._work_trigger`` asks ``may_post``)."""
        if self.manager is None:
            return []
        try:
            results = self.manager.run_due() or []
        except Exception:  # noqa: BLE001
            return []
        out = []
        for r in results:
            to_dict = getattr(r, "to_dict", None)
            if isinstance(r, dict):
                out.append(dict(r))
            elif callable(to_dict):
                out.append(dict(to_dict()))
            else:
                out.append({"status": getattr(r, "status", "posted"),
                            "value": str(r)})
        return out

    def draft_for(self, action: str) -> dict[str, Any]:
        """Draft (never send) for a social action."""
        return {"action": action, "text": action,
                "status": "draft", "sent": False}


# ── Leads ──────────────────────────────────────────────────────────

class LeadsAdapter:
    """Real ``social/leads.py`` capture / qualification / funnel."""

    def __init__(self, store: Any | None = None) -> None:
        self.store = store

    @staticmethod
    def _lead_dict(lead: Any) -> dict[str, Any]:
        if isinstance(lead, dict):
            return dict(lead)
        to_dict = getattr(lead, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return {"lead_id": getattr(lead, "lead_id", str(lead)),
                "name": getattr(lead, "name", ""),
                "platform": getattr(lead, "platform", ""),
                "contact": getattr(lead, "contact", ""),
                "status": getattr(lead, "status", "new")}

    def live_leads(self) -> list[dict[str, Any]]:
        """Every lead that is not dead — each keeps next action."""
        if self.store is None:
            return []
        try:
            leads = self.store.list_leads() or []
        except Exception:  # noqa: BLE001
            return []
        return [self._lead_dict(l) for l in leads
                if self._lead_dict(l).get("status") != "dead"]

    def capture(self, *, name: str, platform: str, contact: str,
                interest: str = "", source_post: str = "") -> dict[str, Any]:
        if self.store is None:
            return {"status": "no_store"}
        try:
            lead = self.store.capture_lead(
                name, platform, contact,
                interest=interest, source_post=source_post)
        except Exception as exc:  # noqa: BLE001
            return {"status": "capture_failed", "error": str(exc)}
        return self._lead_dict(lead) if lead is not None else {}

    def set_store_status(self, lead_id: str, store_status: str) -> bool:
        if self.store is None:
            return False
        try:
            return bool(self.store.set_lead_status(lead_id, store_status))
        except Exception:  # noqa: BLE001
            return False

    def advance(self, lead: dict[str, Any], nxt: str) -> dict[str, Any]:
        """Validate the lifecycle transition, mirror it to the store."""
        current = str(lead.get("stage", lead.get("status", "discovered")))
        try:
            validated = advance_lead(current, nxt)
        except ValueError as exc:
            return {"lead_id": lead.get("lead_id", ""), "status": "invalid",
                    "error": str(exc)}
        lead_id = str(lead.get("lead_id", ""))
        store_status = LEAD_STORE_MAP.get(validated, "new")
        mirrored = self.set_store_status(lead_id, store_status) \
            if lead_id else False
        return {"lead_id": lead_id, "from": current, "to": validated,
                "store_status": store_status, "mirrored": mirrored,
                "next_action": f"lead {lead_id} now {validated}",
                "status": "advanced"}

    def follow_up(self, lead: dict[str, Any], *,
                  draft: str = "") -> dict[str, Any]:
        """Follow-up: store status + a DM DRAFT (never sent here)."""
        lead_id = str(lead.get("lead_id", ""))
        mirrored = self.set_store_status(lead_id, "followed_up") \
            if lead_id else False
        return {"lead_id": lead_id, "status": "followed_up",
                "mirrored": mirrored,
                "draft": draft or f"follow-up for {lead.get('name', lead_id)}",
                "sent": False}


# ── Gigs ───────────────────────────────────────────────────────────

class GigsAdapter:
    """Real ``OpportunityHunter`` + ``GigApplier``.

    Submit honesty: ``submit()`` passes ``explicit`` through to the
    applier AND requires standing authority for that gig. Anything
    else returns ``submit_attempted`` with no evidence — never a
    fabricated "submitted".
    """

    def __init__(self, hunter: Any | None = None,
                 applier: Any | None = None,
                 payout_extractor: Any | None = None,
                 opp_factory: Any | None = None) -> None:
        self.hunter = hunter
        self.applier = applier
        #: ``(payout_text) -> float | None`` — injected from
        #: ``agents.opportunities.extract_payout_usd`` by the caller
        #: (higher layer). Without it, payout defaults to 0 with
        #: uncertainty preserved (never an invented payout).
        self.payout_extractor = payout_extractor
        #: ``(opp_dict) -> Opportunity`` — injected
        #: ``Opportunity.from_dict`` by the caller.
        self.opp_factory = opp_factory

    @staticmethod
    def _opp_dict(opp: Any) -> dict[str, Any]:
        if isinstance(opp, dict):
            return dict(opp)
        to_dict = getattr(opp, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else {"id": str(opp)}

    def scan(self) -> list[dict[str, Any]]:
        if self.hunter is None:
            return []
        try:
            return [self._opp_dict(o) for o in (self.hunter.scan() or [])]
        except Exception:  # noqa: BLE001
            return []

    def score(self, opp: dict[str, Any]) -> dict[str, Any]:
        """NET expected earnings, uncertainty preserved."""
        payout = float(opp.get("payout_usd", 0) or 0)
        if not payout and self.payout_extractor is not None:
            try:
                payout = float(self.payout_extractor(
                    opp.get("payout_text", "") or "") or 0.0)
            except Exception:  # noqa: BLE001
                payout = 0.0
        return score_gig(advertised_payout=payout, win_probability=float(
            opp.get("win_probability", 0.3) or 0.3),
            costs=float(opp.get("costs", 0) or 0),
            hours=float(opp.get("hours", 1) or 1),
            scam_indicators=list(opp.get("scam_indicators") or []))

    def draft(self, opp: dict[str, Any]) -> dict[str, Any]:
        if self.applier is None:
            return {"gig_id": opp.get("id", ""), "status": "no_applier"}
        try:
            if isinstance(opp, dict):
                if self.opp_factory is None:
                    return {"gig_id": opp.get("id", ""),
                            "status": "draft_failed",
                            "error": "no Opportunity factory injected — "
                                     "pass real Opportunity objects"}
                real_opp = self.opp_factory(opp)
            else:
                real_opp = opp
            app = self.applier.draft(real_opp)
        except Exception as exc:  # noqa: BLE001
            return {"gig_id": opp.get("id", ""), "status": "draft_failed",
                    "error": str(exc)}
        to_dict = getattr(app, "to_dict", None)
        if isinstance(app, dict):
            out = dict(app)
        elif callable(to_dict):
            out = dict(to_dict())
        else:
            out = {"gig_id": opp.get("id", ""),
                   "status": getattr(app, "status", "drafted")}
        return out

    def submit(self, gig_id: str, *, explicit: bool = False,
               authority: Any | None = None) -> dict[str, Any]:
        """Submit through the EXISTING authority systems.

        ``authority`` is an :class:`AgencyAuthority` (policy gradient
        + trial-pattern explicit go-ahead). Without an allowed
        decision the result is ``submit_attempted`` with no evidence —
        never a fabricated "submitted". A gradient proposal
        (``proposal_id``) is surfaced so the owner can approve it.
        """
        if authority is not None:
            decision = authority.may_submit(gig_id, explicit=explicit)
            if not decision.allowed:
                return {"gig_id": gig_id, "status": "submit_attempted",
                        "evidence": None, "reason": decision.reason,
                        "proposal_id": getattr(
                            decision, "proposal_id", "") or None}
        elif not explicit:
            return {"gig_id": gig_id, "status": "submit_attempted",
                    "evidence": None,
                    "reason": "submit needs explicit owner instruction "
                              "for THAT gig (trial pattern)"}
        if self.applier is None:
            return {"gig_id": gig_id, "status": "submit_attempted",
                    "evidence": None, "reason": "no applier injected"}
        try:
            app = self.applier.submit(gig_id, explicit=True)
        except Exception as exc:  # noqa: BLE001
            return {"gig_id": gig_id, "status": "submit_attempted",
                    "evidence": None, "error": str(exc)}
        to_dict = getattr(app, "to_dict", None)
        app_d = dict(app) if isinstance(app, dict) else (
            dict(to_dict()) if callable(to_dict) else {})
        # Honesty: "submitted" requires the REAL submission_evidence
        # field — checking any other key would downgrade (or fake)
        # every real submission.
        evidence = app_d.get("submission_evidence") or {}
        status = app_d.get("status", "")
        if status == "submitted" and evidence:
            return {"gig_id": gig_id, "status": "submitted",
                    "evidence": evidence}
        return {"gig_id": gig_id, "status": "submit_attempted",
                "evidence": None,
                "reason": (f"applier returned {status or 'no status'} "
                           "without submission evidence"),
                "applier_status": status}

    def follow_ups_due(self) -> list[dict[str, Any]]:
        if self.applier is None:
            return []
        try:
            apps = self.applier.follow_ups_due() or []
        except Exception:  # noqa: BLE001
            return []
        return [self._opp_dict(a) for a in apps]

    def follow_up_draft(self, gig_id: str) -> dict[str, Any]:
        if self.applier is None:
            return {"gig_id": gig_id, "status": "no_applier"}
        try:
            out = self.applier.follow_up_draft(gig_id)
        except Exception as exc:  # noqa: BLE001
            return {"gig_id": gig_id, "status": "draft_failed",
                    "error": str(exc)}
        if isinstance(out, dict):
            return dict(out)
        return {"gig_id": gig_id, "draft": str(out), "status": "drafted"}

    def set_status(self, gig_id: str, status: str) -> dict[str, Any]:
        if self.applier is None:
            return {"gig_id": gig_id, "status": "no_applier"}
        try:
            out = self.applier.set_status(gig_id, status)
        except Exception as exc:  # noqa: BLE001
            return {"gig_id": gig_id, "status": "set_failed",
                    "error": str(exc)}
        if isinstance(out, dict):
            return dict(out)
        to_dict = getattr(out, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return {"gig_id": gig_id, "status": status}


# ── Planner ────────────────────────────────────────────────────────

class PlannerAdapter:
    """Real ``planning/graph_missions.py`` plan-graph execution.

    The graph functions (``plan_authoring``, ``PlanGraph.from_dict``,
    ``execute_graph``) are INJECTED by the caller (higher layer) —
    the agency layer never imports ``planning`` (layering). Without
    them, graphs are reported unexecutable, never faked.
    """

    def __init__(self, graphs: list[dict[str, Any]] | None = None,
                 actions: dict[str, Any] | None = None,
                 execute_fn: Any | None = None,
                 compile_fn: Any | None = None,
                 graph_factory: Any | None = None) -> None:
        self._graphs = list(graphs or [])
        self.actions = dict(actions or {})
        self.execute_fn = execute_fn
        self.compile_fn = compile_fn
        self.graph_factory = graph_factory

    def pending_graphs(self) -> list[dict[str, Any]]:
        return list(self._graphs)

    def compile(self, spec: dict[str, Any]) -> dict[str, Any]:
        if self.compile_fn is None:
            return {"ok": False, "error": "no plan compiler injected"}
        graph = self.compile_fn(spec)
        to_dict = getattr(graph, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else {"ok": bool(graph)}

    def execute_graph(self, graph: Any) -> dict[str, Any]:
        real = graph
        if isinstance(graph, dict):
            if self.graph_factory is None:
                return {"completed": [], "failed": ["unparseable graph"],
                        "attempts": 0,
                        "error": "no graph factory injected"}
            real = self.graph_factory(graph)
        if real is None:
            return {"completed": [], "failed": ["unparseable graph"],
                    "attempts": 0}
        if self.execute_fn is None:
            return {"completed": [], "failed": ["no plan executor injected"],
                    "attempts": 0}
        result = self.execute_fn(real, actions=self.actions)
        # A plan graph is work to run ONCE: executing consumes it, so
        # the next tick moves to the next graph instead of re-doing
        # completed work.
        try:
            if graph in self._graphs:
                self._graphs.remove(graph)
        except Exception:  # noqa: BLE001 - consumption is best-effort
            pass
        to_dict = getattr(result, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else {
            "completed": list(getattr(result, "completed", [])),
            "failed": list(getattr(result, "failed", [])),
            "attempts": getattr(result, "attempts", 0)}


# ── assembly (NO construction here) ──────────────────────────────

def build_adapters(*, hunter: Any | None = None,
                   weather: Any | None = None,
                   news_agent: Any | None = None,
                   social_manager: Any | None = None,
                   leads_store: Any | None = None,
                   opp_hunter: Any | None = None,
                   applier: Any | None = None,
                   payout_extractor: Any | None = None,
                   opp_factory: Any | None = None,
                   planner_graphs: list[dict[str, Any]] | None = None,
                   planner_actions: dict[str, Any] | None = None,
                   graph_execute_fn: Any | None = None,
                   graph_compile_fn: Any | None = None,
                   graph_factory: Any | None = None,
                   places: list[str] | None = None,
                   inbox: Any | None = None) -> dict[str, Any]:
    """Assemble adapters from INJECTED organs. Never raises, never imports.

    Layering (ARCHITECTURE.md §1): the agency layer is L0 — it must
    not import any organ (``core``/``agents``/``social``/``planning``/
    ``integrations``/``finance``). Organs are constructed by the
    CALLER (a higher layer: ``tools/autonomy.py``'s shift tool,
    ``agents/partner/runtime.py``) and passed in here. An adapter
    without its organ returns empty observations and refuses actions
    — it never fabricates organ output.
    """
    return {
        "deals": DealsAdapter(hunter=hunter),
        "weather": WeatherAdapter(weather=weather,
                                  places=list(places or [])),
        "news": NewsAdapter(agent=news_agent),
        "social": SocialAdapter(manager=social_manager, inbox=inbox),
        "leads": LeadsAdapter(store=leads_store),
        "gigs": GigsAdapter(hunter=opp_hunter, applier=applier,
                            payout_extractor=payout_extractor,
                            opp_factory=opp_factory),
        "planner": PlannerAdapter(graphs=planner_graphs,
                                  actions=planner_actions,
                                  execute_fn=graph_execute_fn,
                                  compile_fn=graph_compile_fn,
                                  graph_factory=graph_factory),
        "_built_at": time.time(),
    }
