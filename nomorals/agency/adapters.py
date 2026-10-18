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
    "LEAD_STORE_MAP", "build_adapters", "default_social_inbox",
    "places_from_settings",
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
    CAP_PURCHASE = "commerce.purchase"

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

    def may_dm(self, *, draft: str = "", explicit: bool = False) -> Any:
        """DM decision. ``social.dm`` is confirmable: autonomous sends
        need a confirmation token (gradient denies without one), while
        ``explicit=True`` is the trial pattern — the owner explicitly
        approved THIS draft, so the confirmation gate is bypassed
        (deny rules, grants, and biometric still apply)."""
        return self.check(self.CAP_DM, draft=draft or "autonomous DM",
                          explicit_override=explicit)

    def may_purchase(self, *, amount: float = 0.0,
                     draft: str = "") -> Any:
        """Purchase decision: policy gate for ``commerce.purchase``.

        This is the POLICY half of a purchase (the gradient / allow
        rules). The MANDATE half is :meth:`spend_decision` — a purchase
        executes only when BOTH allow (see
        :meth:`DealsAdapter.execute_purchase`).
        """
        return self.check(self.CAP_PURCHASE,
                          draft=draft or f"purchase of {amount:.2f}",
                          context={"amount": amount})

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
    organizing axis. ``check_need`` verifies and reports readiness /
    blockers; ``execute_purchase`` COMPLETES a purchase through the
    injected ``purchase_executor`` — but only when standing authority
    allows (policy gate AND payment mandate). Without authority, or
    without an executor, it returns a blocker and buys nothing."""

    def __init__(self, hunter: Any | None = None,
                 price_check: Any | None = None,
                 purchase_executor: Any | None = None) -> None:
        self.hunter = hunter
        self.price_check = price_check
        #: ``(need, observed) -> dict`` — the real checkout channel.
        #: Must return ``{"purchased": True, "receipt": {...}}`` on
        #: success; anything without a receipt is NOT a purchase.
        #: Injected by the caller (higher layer); None = no channel.
        self.purchase_executor = purchase_executor

    def execute_purchase(self, need: dict[str, Any],
                         observed: dict[str, Any], *,
                         authority: Any | None = None) -> dict[str, Any]:
        """Complete a purchase for a verified saving, when authorized.

        Gates, in order: (1) policy — ``authority.may_purchase``;
        (2) mandate — ``authority.can_spend(amount)``; (3) executor —
        a checkout channel must be injected. Every denial returns
        ``{"purchased": False, ...}`` with the reason — never a
        silent buy, never a fabricated receipt. A receipt is required:
        an executor that reports success without receipt evidence is
        treated as NOT purchased.
        """
        amount = float(observed.get("landed_price",
                                    observed.get("price", 0)) or 0)
        if authority is not None:
            may = getattr(authority, "may_purchase", None)
            if callable(may):
                decision = may(amount=amount, draft=(
                    f"purchase {need.get('product', '')} "
                    f"for {amount:.2f}"))
                if not getattr(decision, "allowed", False):
                    return {"purchased": False,
                            "reason": getattr(
                                decision, "reason",
                                "purchase not allowed by policy"),
                            "proposal_id": getattr(
                                decision, "proposal_id", "") or None,
                            "blocker": "purchase blocked by policy"}
            if not authority.can_spend(amount):
                return {"purchased": False,
                        "reason": authority.spend_reason(amount),
                        "blocker": "purchase blocked: no payment mandate"}
        elif self.purchase_executor is not None:
            return {"purchased": False,
                    "reason": "no authority configured — cannot spend",
                    "blocker": "purchase blocked: no authority"}
        if self.purchase_executor is None:
            return {"purchased": False,
                    "reason": "no purchase executor injected — no "
                              "checkout channel configured for this vendor",
                    "blocker": "purchase blocked: no purchase executor"}
        try:
            result = self.purchase_executor(dict(need), dict(observed))
        except Exception as exc:  # noqa: BLE001 - executor failure = no buy
            return {"purchased": False, "error": str(exc),
                    "blocker": f"purchase executor failed: {exc}"}
        if not isinstance(result, dict) or not result.get("purchased"):
            return {"purchased": False,
                    "reason": str((result or {}).get(
                        "reason", "executor did not complete a purchase")),
                    "blocker": "purchase not completed by executor",
                    "executor_result": result}
        receipt = result.get("receipt")
        if not receipt:
            return {"purchased": False,
                    "reason": "executor reported success without receipt "
                              "evidence — treated as not purchased",
                    "blocker": "purchase blocked: no receipt evidence",
                    "executor_result": result}
        return {"purchased": True, "receipt": receipt,
                "amount": amount, "product": need.get("product", "")}

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

def _as_place_list(value: Any) -> list[str]:
    """Coerce a settings place value to a list of place-name strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        out: list[str] = []
        for key in ("home_place", "work_place", "location", "places"):
            out.extend(_as_place_list(value.get(key)))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_as_place_list(item))
        return out
    label = getattr(value, "name", None) or getattr(value, "label", None)
    if isinstance(label, str) and label.strip():
        return [label.strip()]
    return []


def places_from_settings(settings: Any) -> list[str]:
    """Default weather places derived from user settings (duck-typed).

    Reads, in order: ``settings.location`` (``places`` + ``home_place``
    + ``work_place``), then ``settings`` itself (``places`` /
    ``home_place`` / ``work_place`` / ``location``). ``settings.home``
    is the DATA home (``~/.nomorals``) — never a place — and is
    deliberately NOT read. Returns [] when nothing is configured, so
    the caller falls through to the next derivation source. Never
    raises, never fabricates a place.
    """
    if settings is None:
        return []
    out: list[str] = []
    location = getattr(settings, "location", None)
    if location is not None and location is not settings:
        out.extend(_as_place_list(getattr(location, "places", None)))
        out.extend(_as_place_list(getattr(location, "home_place", None)))
        out.extend(_as_place_list(getattr(location, "work_place", None)))
    out.extend(_as_place_list(getattr(settings, "places", None)))
    out.extend(_as_place_list(getattr(settings, "home_place", None)))
    out.extend(_as_place_list(getattr(settings, "work_place", None)))
    loc = getattr(settings, "location", None)
    if isinstance(loc, (str, list, tuple)):
        out.extend(_as_place_list(loc))
    seen: set[str] = set()
    deduped: list[str] = []
    for place in out:
        if place and place not in seen:
            seen.add(place)
            deduped.append(place)
    return deduped


class WeatherAdapter:
    """Real ``agents/weather.py`` forecasts + official warnings.

    Default places derive from user settings (home/work/location
    config) when not explicitly configured — basic function never
    requires manual config. Derivation order: explicit ``places`` →
    ``settings`` (home/work/location) → ``places_provider`` callable
    → the organ's own ``default_place`` → [] (no source at all).
    """

    def __init__(self, weather: Any | None = None,
                 places: list[str] | None = None,
                 settings: Any | None = None,
                 places_provider: Any | None = None) -> None:
        self._weather = weather
        self._places = list(places or [])
        self._settings = settings
        self._places_provider = places_provider

    def _organ(self) -> Any:
        # The organ is INJECTED by the caller (higher layer) — the
        # agency layer never imports it (layering). No organ = no
        # observations, never fabricated weather.
        return self._weather

    def places(self) -> list[str]:
        if self._places:
            return list(self._places)
        derived = places_from_settings(self._settings)
        if derived:
            return derived
        if self._places_provider is not None:
            try:
                provided = self._places_provider()
            except Exception:  # noqa: BLE001 - provider failure = no places
                provided = None
            if provided:
                return _as_place_list(provided)
        organ = self._organ()
        default = getattr(organ, "default_place", None) if organ else None
        if isinstance(default, str) and default.strip():
            return [default.strip()]
        return []

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

def _iter_relationships(relationships: Any) -> list[Any]:
    """Relationships from any tracker shape (duck-typed, never raises)."""
    if relationships is None:
        return []
    try:
        if callable(getattr(relationships, "all", None)):
            return list(relationships.all() or [])
        if callable(getattr(relationships, "top_contacts", None)):
            return list(relationships.top_contacts(1000) or [])
        store = getattr(relationships, "_rels", None)
        if isinstance(store, dict):
            return list(store.values())
        if isinstance(relationships, (list, tuple)):
            return list(relationships)
    except Exception:  # noqa: BLE001 - tracker failure = no relationships
        pass
    return []


def default_social_inbox(*, manager: Any | None = None,
                         gateway: Any | None = None,
                         relationships: Any | None = None,
                         draft_queue: Any | None = None) -> Any:
    """Default DM/triage source from the REAL social organs.

    No injected ``inbox`` needed: the backlog is derived from
    actually-wired state —
    * **unanswered** — relationship-tracker contacts whose
      ``last_inbound`` is newer than ``last_outbound`` (they messaged
      last and are still waiting), plus triage-log entries needing
      attention when the manager exposes one;
    * **promised_replies** — pending drafts in the real
      ``DraftQueue`` (a draft held for approval IS a promised reply);
    * **milestones** — contacts due for reconnect per the tracker
      (``due_for_reconnect``), i.e. relationship milestones to honour;
    * **stop_signal / cadence_ok** — from the gateway when it exposes
      them, else the honest defaults (no signal, cadence open).

    Returns a zero-arg callable with the ``backlog()`` contract, so it
    drops into :class:`SocialAdapter` exactly like an injected inbox.
    Never raises; an organ that exposes nothing contributes nothing.
    """
    def _inbox() -> dict[str, Any]:
        unanswered = 0
        milestones = 0
        try:
            for rel in _iter_relationships(relationships):
                last_in = float(getattr(rel, "last_inbound", 0) or 0)
                last_out = float(getattr(rel, "last_outbound", 0) or 0)
                if last_in > last_out:
                    unanswered += 1
        except Exception:  # noqa: BLE001 - never raises
            pass
        try:
            due = getattr(relationships, "due_for_reconnect", None)
            if callable(due):
                milestones += len(due() or [])
        except Exception:  # noqa: BLE001
            pass
        promised = 0
        queue = draft_queue
        if queue is None and manager is not None:
            queue = getattr(manager, "draft_queue", None) or getattr(
                manager, "drafts", None)
        try:
            if queue is not None and callable(getattr(queue, "pending", None)):
                promised += len(queue.pending() or [])
        except Exception:  # noqa: BLE001
            pass
        if manager is not None:
            try:
                triage = getattr(manager, "triage_log", None)
                if triage is not None and callable(
                        getattr(triage, "digest", None)):
                    digest = triage.digest() or {}
                    needs = digest.get("needs_attention") or []
                    unanswered += len(needs)
            except Exception:  # noqa: BLE001
                pass
        stop_signal = False
        cadence_ok = True
        if gateway is not None:
            try:
                stop_signal = bool(getattr(gateway, "stop_signal", False))
                cadence = getattr(gateway, "cadence_ok", None)
                if cadence is not None:
                    cadence_ok = bool(cadence() if callable(cadence)
                                      else cadence)
            except Exception:  # noqa: BLE001
                pass
        return {"unanswered": unanswered, "promised_replies": promised,
                "milestones": milestones, "stop_signal": stop_signal,
                "cadence_ok": cadence_ok}

    return _inbox


class SocialAdapter:
    """Real social executors (posting, community, DM handling).

    ``manager`` is the real ``social/manager.py`` SocialManager
    (``due()`` / ``run_due()``). ``inbox`` is an injected callable
    returning the unanswered / promised / milestone backlog
    (``{"unanswered", "promised_replies", "milestones",
    "stop_signal", "cadence_ok"}``). When no inbox is injected, the
    default is derived from the real organs
    (:func:`default_social_inbox` — relationships tracker, draft
    queue, triage log), never fabricated. Drafts are built, never
    sent, without authority.
    """

    def __init__(self, manager: Any | None = None,
                 inbox: Any | None = None,
                 gateway: Any | None = None,
                 relationships: Any | None = None,
                 draft_queue: Any | None = None) -> None:
        self.manager = manager
        self.gateway = gateway
        self.relationships = relationships
        self.draft_queue = draft_queue
        if inbox is not None:
            self.inbox = inbox
        else:
            self.inbox = default_social_inbox(
                manager=manager, gateway=gateway,
                relationships=relationships, draft_queue=draft_queue)

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

    def __init__(self, store: Any | None = None,
                 dm_sender: Any | None = None) -> None:
        self.store = store
        #: ``(platform, contact, body) -> bool`` — the real DM-send
        #: channel (the chat gateway's send path, injected by the
        #: caller). None = no DM channel configured: sends block.
        self.dm_sender = dm_sender

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

    def send_dm(self, lead: dict[str, Any], *,
                draft: str = "", authority: Any | None = None,
                explicit: bool = False) -> dict[str, Any]:
        """Send the owner-approved DM draft for a lead, when authorized.

        The contact step's executor. Gates: (1) ``authority.may_dm``
        when an authority is given — ``explicit=True`` means the
        owner approved THIS draft (trial pattern: the confirmation
        gate is bypassed, deny/grant/biometric still apply); without
        it, a confirmable capability denies and the draft stays held;
        (2) a ``dm_sender`` channel must be injected. On a successful
        send the store status moves to ``dmed`` and the result
        carries the sent evidence (platform / contact / draft) —
        never a bare claim.
        """
        lead_id = str(lead.get("lead_id", ""))
        platform = str(lead.get("platform", ""))
        contact = str(lead.get("contact", ""))
        text = draft or f"Hi {lead.get('name', '') or lead_id} — following up."
        if authority is not None:
            may_dm = getattr(authority, "may_dm", None)
            if callable(may_dm):
                try:
                    decision = may_dm(draft=text, explicit=explicit)
                except TypeError:
                    decision = may_dm(draft=text)
                if not getattr(decision, "allowed", False):
                    return {"lead_id": lead_id, "sent": False,
                            "reason": getattr(
                                decision, "reason",
                                "DM not allowed by authority"),
                            "proposal_id": getattr(
                                decision, "proposal_id", "") or None,
                            "draft": text}
        if self.dm_sender is None:
            return {"lead_id": lead_id, "sent": False,
                    "reason": "no DM sender injected — no DM channel "
                              "configured",
                    "draft": text}
        if not contact:
            return {"lead_id": lead_id, "sent": False,
                    "reason": "lead has no contact to DM",
                    "draft": text}
        try:
            sent = bool(self.dm_sender(platform, contact, text))
        except Exception as exc:  # noqa: BLE001 - sender failure = not sent
            return {"lead_id": lead_id, "sent": False,
                    "reason": f"DM sender failed: {exc}", "draft": text}
        if not sent:
            return {"lead_id": lead_id, "sent": False,
                    "reason": "DM sender reported failure", "draft": text}
        mirrored = self.set_store_status(lead_id, "dmed") if lead_id else False
        return {"lead_id": lead_id, "sent": True, "mirrored": mirrored,
                "evidence": {"platform": platform, "contact": contact,
                             "draft": text}}


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
                   inbox: Any | None = None,
                   settings: Any | None = None,
                   places_provider: Any | None = None,
                   gateway: Any | None = None,
                   relationships: Any | None = None,
                   draft_queue: Any | None = None,
                   dm_sender: Any | None = None,
                   purchase_executor: Any | None = None) -> dict[str, Any]:
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
        "deals": DealsAdapter(hunter=hunter,
                              purchase_executor=purchase_executor),
        "weather": WeatherAdapter(weather=weather,
                                  places=list(places or []),
                                  settings=settings,
                                  places_provider=places_provider),
        "news": NewsAdapter(agent=news_agent),
        "social": SocialAdapter(manager=social_manager, inbox=inbox,
                                gateway=gateway,
                                relationships=relationships,
                                draft_queue=draft_queue),
        "leads": LeadsAdapter(store=leads_store, dm_sender=dm_sender),
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
