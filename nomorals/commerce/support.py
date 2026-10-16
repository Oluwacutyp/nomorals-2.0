"""Support inbox: WhatsApp-native tickets with SLA.

Business support runs where the customers are — WhatsApp first. A ticket
is opened from any channel (WhatsApp, email, in-store, web), assigned to
an agent, answered with a *draft* (a human sends it, or an approved
auto-sender does — this module never sends anything itself), and closed
with a resolution.

SLA targets (first-response, from ticket creation):
- urgent → 1 hour · high → 4 hours · medium → 24 hours · low → 72 hours

``sla_breach`` is computed from the ticket's own clock — no invented
"we replied fast" claims. ``reply_draft`` produces suggested wording
for the agent; it is a draft, not a sent message.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Ticket",
    "SLA_TARGETS_S",
    "SupportInbox",
]

#: First-response SLA targets in seconds, by priority.
SLA_TARGETS_S: dict[str, int] = {
    "urgent": 1 * 3600,
    "high": 4 * 3600,
    "medium": 24 * 3600,
    "low": 72 * 3600,
}

STATUSES: tuple[str, ...] = ("open", "assigned", "awaiting_customer",
                             "resolved", "closed")
PRIORITIES: tuple[str, ...] = tuple(SLA_TARGETS_S)

_REPLY_TEMPLATES: dict[str, str] = {
    "greeting": ("Hi {name}, thanks for reaching out to {store} — "
                 "I'm looking into this for you now."),
    "order_status": ("Hi {name}, your order ({reference}) is {status}. "
                     "{detail}"),
    "refund": ("Hi {name}, sorry about this. I've logged your refund "
               "request for {reference} — you'll hear back within "
               "{sla}."),
    "delivery": ("Hi {name}, your delivery ({reference}) is on its way. "
                 "I'll share the tracking details as soon as the courier "
                 "confirms pickup."),
    "closing": ("Hi {name}, glad we could sort this out! Reply here if "
                "anything else comes up — we're happy to help. 🙏"),
}


@dataclass
class Ticket:
    id: str
    channel: str = "whatsapp"  # whatsapp | email | in_store | web
    customer: str = ""
    customer_phone: str = ""
    message: str = ""
    status: str = "open"
    priority: str = "medium"
    assignee: str = ""
    store: str = ""
    created_at: float = 0.0
    first_response_at: float = 0.0
    closed_at: float = 0.0
    resolution: str = ""
    replies: list[dict[str, Any]] = field(default_factory=list)

    def sla_deadline(self) -> float:
        return self.created_at + SLA_TARGETS_S.get(self.priority, 24 * 3600)

    def sla_breached(self, now: float | None = None) -> bool:
        """True when first response missed the SLA (or is still missing past it)."""
        moment = time.time() if now is None else now
        if self.first_response_at:
            return self.first_response_at > self.sla_deadline()
        return moment > self.sla_deadline() and self.status not in ("closed", "resolved")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "channel": self.channel, "customer": self.customer,
            "customer_phone": self.customer_phone, "message": self.message,
            "status": self.status, "priority": self.priority,
            "assignee": self.assignee, "store": self.store,
            "created_at": self.created_at,
            "first_response_at": self.first_response_at,
            "closed_at": self.closed_at, "resolution": self.resolution,
            "replies": list(self.replies),
        }


class SupportInbox:
    """In-memory support inbox. ``now`` injectable for tests."""

    def __init__(self, *, now: Callable[[], float] | None = None) -> None:
        self._now = now or time.time
        self.tickets: list[Ticket] = []

    # ── intake ──

    def open(self, customer: str, message: str, *,
             channel: str = "whatsapp", customer_phone: str = "",
             priority: str = "medium", store: str = "") -> Ticket:
        if not (customer or "").strip() and not (customer_phone or "").strip():
            raise ValueError("ticket needs a customer name or phone")
        if not (message or "").strip():
            raise ValueError("ticket message is required")
        if priority not in PRIORITIES:
            raise ValueError(f"unknown priority {priority!r} — {PRIORITIES}")
        ticket = Ticket(
            id="tkt_" + uuid.uuid4().hex[:10],
            channel=channel or "whatsapp", customer=customer,
            customer_phone=customer_phone, message=message,
            priority=priority, store=store, created_at=self._now(),
        )
        self.tickets.append(ticket)
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        return next((t for t in self.tickets if t.id == ticket_id), None)

    # ── workflow ──

    def assign(self, ticket: Ticket, assignee: str) -> Ticket:
        if not (assignee or "").strip():
            raise ValueError("assignee is required")
        if ticket.status in ("closed", "resolved"):
            raise ValueError(f"cannot assign a {ticket.status} ticket")
        ticket.assignee = assignee
        ticket.status = "assigned"
        return ticket

    def reply_draft(self, ticket: Ticket, template: str = "greeting", *,
                    store: str = "", reference: str = "",
                    detail: str = "", status_text: str = "") -> dict[str, Any]:
        """Build a reply DRAFT. Never sends. Records first-response time.

        The first draft counts as the first response for SLA purposes —
        the draft is what the agent sends, so its timestamp is honest.
        """
        if template not in _REPLY_TEMPLATES:
            raise ValueError(
                f"unknown template {template!r} — {sorted(_REPLY_TEMPLATES)}")
        if ticket.status in ("closed",):
            raise ValueError("cannot reply to a closed ticket")
        name = ticket.customer.strip() or "there"
        sla_hours = SLA_TARGETS_S.get(ticket.priority, 86400) // 3600
        text = _REPLY_TEMPLATES[template].format(
            name=name, store=store or ticket.store or "our store",
            reference=reference or ticket.id,
            status=status_text or "being processed",
            detail=detail, sla=f"{sla_hours}h")
        if not ticket.first_response_at:
            ticket.first_response_at = self._now()
        draft = {"ticket_id": ticket.id, "template": template,
                 "text": text, "draft": True, "sent": False,
                 "created_at": self._now()}
        ticket.replies.append(draft)
        if ticket.status == "open":
            ticket.status = "assigned" if ticket.assignee else "open"
        return draft

    def close(self, ticket: Ticket, resolution: str = "") -> Ticket:
        if ticket.status == "closed":
            return ticket
        ticket.status = "closed"
        ticket.resolution = resolution
        ticket.closed_at = self._now()
        return ticket

    # ── reporting ──

    def open_tickets(self) -> list[Ticket]:
        return [t for t in self.tickets
                if t.status in ("open", "assigned", "awaiting_customer")]

    def breached(self, now: float | None = None) -> list[Ticket]:
        moment = self._now() if now is None else now
        return [t for t in self.tickets if t.sla_breached(moment)]

    def stats(self, now: float | None = None) -> dict[str, Any]:
        moment = self._now() if now is None else now
        open_n = len(self.open_tickets())
        closed = [t for t in self.tickets if t.status == "closed"]
        responded = [t for t in self.tickets if t.first_response_at]
        avg_first_h = 0.0
        if responded:
            avg_first_h = sum(
                (t.first_response_at - t.created_at) for t in responded
            ) / len(responded) / 3600
        return {
            "total": len(self.tickets),
            "open": open_n,
            "closed": len(closed),
            "sla_breached": len(self.breached(moment)),
            "avg_first_response_hours": round(avg_first_h, 2),
            "by_channel": {c: sum(1 for t in self.tickets if t.channel == c)
                           for c in sorted({t.channel for t in self.tickets})},
            "by_priority": {p: sum(1 for t in self.tickets if t.priority == p)
                            for p in PRIORITIES},
        }
