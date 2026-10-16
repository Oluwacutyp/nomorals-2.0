"""Planning -> commerce logistics: plan steps that book logistics.

A plan (see :mod:`nomorals.planning.graph`) can include a logistics
step: "ship the parcel / move the stock / deliver the order".  This
module turns a :class:`LogisticsDescriptor` into a costed, ETA'd plan
step.

Layering note (why there is no ``commerce`` import here): planning is
L3, commerce is L4 — planning must not import commerce.  The real
carrier quotes therefore arrive through *injected* callables
(``quote_fn`` / ``eta_fn``), which an L4+ caller adapts from
``nomorals.commerce.logistics.rate_quote`` / ``eta``::

    from nomorals.commerce.logistics import Address, eta, rate_quote
    from nomorals.planning.commerce_logistics import (
        LogisticsDescriptor, plan_logistics_step)

    def quote_fn(d):  # d is a LogisticsDescriptor
        return rate_quote("gig", origin=Address(...), destination=Address(...),
                          weight_kg=d.weight_kg, distance_km=d.distance_km)

    step = plan_logistics_step("ship order", descriptor, quote_fn=quote_fn)

Without injected callables the module falls back to a generic offline
planning estimate — clearly labelled ``estimated=True`` with
``source="planning fallback"`` — never presented as a carrier quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "LogisticsDescriptor",
    "LogisticsPlanStep",
    "estimate_cost",
    "estimate_eta",
    "plan_logistics_step",
    "attach_logistics_step",
    "logistics_step_to_dict",
    "FALLBACK_BASE_MINOR",
    "FALLBACK_PER_KM_MINOR",
    "FALLBACK_PER_KG_MINOR",
]

#: Generic planning-fallback tariff (minor units, NGN kobo).  These are
#: NOT any carrier's prices — they exist so a plan can carry a costed
#: logistics step before a real quote is wired in.
FALLBACK_BASE_MINOR = 150_000
FALLBACK_PER_KM_MINOR = 60
FALLBACK_PER_KG_MINOR = 40_000


@dataclass
class LogisticsDescriptor:
    """What needs moving. Origin/destination are free-text place names
    (``"Lagos, Ikeja"``) — the injected quote adapter resolves them to
    carrier addresses; the planner itself never geocodes."""

    carrier: str = "gig"
    origin: str = ""
    destination: str = ""
    weight_kg: float = 1.0
    distance_km: float | None = None
    currency: str = "NGN"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier": self.carrier,
            "origin": self.origin,
            "destination": self.destination,
            "weight_kg": self.weight_kg,
            "distance_km": self.distance_km,
            "currency": self.currency,
            "notes": self.notes,
        }


@dataclass
class LogisticsPlanStep:
    """A plan step that books logistics, with cost + ETA attached."""

    name: str = ""
    carrier: str = ""
    cost_minor: int = 0
    currency: str = "NGN"
    eta: str = ""
    estimated: bool = True
    source: str = ""
    descriptor: LogisticsDescriptor = field(
        default_factory=LogisticsDescriptor)
    quote: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "logistics",
            "name": self.name,
            "carrier": self.carrier,
            "cost_minor": self.cost_minor,
            "currency": self.currency,
            "eta": self.eta,
            "estimated": self.estimated,
            "source": self.source,
            "descriptor": self.descriptor.to_dict(),
            "quote": dict(self.quote),
        }

    def format(self) -> str:
        """One-line plan card: 'Ship order — GIG, ₦2,500 est, 1–2 days'."""
        cost = f"₦{self.cost_minor / 100:,.0f}" if self.currency == "NGN" \
            else f"{self.cost_minor} {self.currency}"
        est = " (est.)" if self.estimated else ""
        return (f"🚚 {self.name} — {self.carrier or 'logistics'}, "
                f"{cost}{est}, ETA {self.eta or 'unknown'}")


def estimate_cost(descriptor: LogisticsDescriptor, *,
                  quote_fn: Callable[[LogisticsDescriptor], Any] | None = None
                  ) -> dict[str, Any]:
    """Cost for a descriptor. With ``quote_fn`` the adapter's result is
    returned verbatim (``estimated=False`` when the adapter says so);
    without one, the generic planning fallback applies."""
    if descriptor.weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if quote_fn is not None:
        result = quote_fn(descriptor)
        if isinstance(result, dict):
            out = dict(result)
            out.setdefault("estimated", False)
            out.setdefault("source", "live carrier quote (injected)")
            return out
        return {"estimated": False, "source": "live carrier quote (injected)",
                "quote": result}
    km = float(descriptor.distance_km) if descriptor.distance_km is not None else 15.0
    fee = (FALLBACK_BASE_MINOR
           + int(FALLBACK_PER_KM_MINOR * km)
           + int(FALLBACK_PER_KG_MINOR * max(0.0, descriptor.weight_kg - 1.0)))
    return {
        "carrier": descriptor.carrier,
        "fee_minor": fee,
        "currency": descriptor.currency,
        "weight_kg": descriptor.weight_kg,
        "distance_km": km,
        "estimated": True,
        "source": "planning fallback — generic estimate, not a carrier quote",
    }


def estimate_eta(descriptor: LogisticsDescriptor, *,
                 eta_fn: Callable[[LogisticsDescriptor], Any] | None = None
                 ) -> dict[str, Any]:
    """ETA for a descriptor. Same injection rule as :func:`estimate_cost`."""
    if eta_fn is not None:
        result = eta_fn(descriptor)
        if isinstance(result, dict):
            out = dict(result)
            out.setdefault("estimated", False)
            out.setdefault("source", "live carrier ETA (injected)")
            return out
        return {"estimated": False, "source": "live carrier ETA (injected)",
                "eta": result}
    km = descriptor.distance_km
    if km is None:
        window = "1–3 days"
    elif km <= 50:
        window = "same-day to next-day"
    elif km <= 500:
        window = "1–2 days"
    else:
        window = "2–4 days"
    return {
        "carrier": descriptor.carrier,
        "eta": window,
        "estimated": True,
        "source": "planning fallback — distance bands, not a carrier ETA",
    }


def plan_logistics_step(name: str, descriptor: LogisticsDescriptor, *,
                        quote_fn: Callable[[LogisticsDescriptor], Any] | None = None,
                        eta_fn: Callable[[LogisticsDescriptor], Any] | None = None
                        ) -> LogisticsPlanStep:
    """Build a costed, ETA'd logistics plan step from a descriptor."""
    cost = estimate_cost(descriptor, quote_fn=quote_fn)
    eta = estimate_eta(descriptor, eta_fn=eta_fn)
    fee = cost.get("fee_minor")
    if fee is None:
        fee = cost.get("cost_minor", 0)
    return LogisticsPlanStep(
        name=name or "ship / deliver",
        carrier=str(cost.get("carrier") or descriptor.carrier),
        cost_minor=int(fee or 0),
        currency=str(cost.get("currency") or descriptor.currency),
        eta=str(eta.get("eta") or ""),
        estimated=bool(cost.get("estimated", True)
                       or eta.get("estimated", True)),
        source=str(cost.get("source") or eta.get("source") or ""),
        descriptor=descriptor,
        quote={"cost": cost, "eta": eta},
    )


def attach_logistics_step(steps: list[dict[str, Any]],
                          step: LogisticsPlanStep) -> list[dict[str, Any]]:
    """Return a new step list with the logistics step appended. The
    input list is not mutated — plans are append-only history."""
    return [*steps, step.to_dict()]


def logistics_step_to_dict(step: LogisticsPlanStep) -> dict[str, Any]:
    """Serialise a logistics plan step (alias for ``step.to_dict()``)."""
    return step.to_dict()
