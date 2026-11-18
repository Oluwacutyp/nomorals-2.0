"""Commerce tool — storefronts + cart recovery as an agent surface.

The commerce package (``nomorals/commerce/``) had real implementations
(Medusa/WooCommerce provisioning, WhatsApp cart recovery) but almost no
reachable surface: provisioning was only callable from one odd corner of
the partner runtime, and cart recovery had no chat/CLI path at all.
This module is that surface.

Actions: stores | store | provision | recovery_stats | recovery_report |
         cart_webhook | run_due_steps | products | page | pay | invoice |
         subscribe | ship | reconcile | ticket | report | tax

Read actions are always safe. ``provision`` creates real infrastructure
(needs a reachable Medusa backend or WooCommerce site) and fails closed
with the reason when the backend isn't there. ``cart_webhook`` only
*records* carts — the 3-step WhatsApp sequence still requires the
customer's explicit opt-in, enforced inside CartRecovery.

The commerce-organ actions (products/page/pay/invoice/subscribe/ship/
reconcile/ticket/report/tax) delegate to the same
``nomorals.commerce.commands.handle_store_commerce`` the /store chat
surface uses, over the durable store at
``~/.nomorals/data/commerce.json``. Money is integer minor units
(kobo). A live ``pay charge`` needs the provider's real secret key and
a real HTTP round-trip — without either it refuses in words.
"""

from __future__ import annotations

import os
from typing import Any

from ..core.logging_setup import get_logger

__all__ = ["register"]

_log = get_logger(__name__)


def _store_manager() -> Any:
    from ..commerce.medusa import StoreManager
    return StoreManager()


def _recovery(db_path: str = "") -> Any:
    from ..commerce.cart_recovery import CartRecovery
    path = db_path or os.path.expanduser(
        "~/.nomorals/commerce/cart_recovery.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return CartRecovery(db_path=path)


def _commerce_tail(action: str, kwargs: dict[str, Any]) -> str:
    """Build a /store-style tail for a commerce-organ action.

    The chat handler is the single implementation; the tool family
    speaks to it in its own words so the two surfaces can never drift.
    """
    sub = str(kwargs.get("sub") or "").strip().lower()
    if action == "products":
        if sub == "add":
            tail = (f"products add {kwargs.get('product_id')} "
                    f"{kwargs.get('name')} {kwargs.get('price_minor')}")
            pipes = [str(kwargs.get("currency") or ""),
                     str(kwargs.get("category") or ""),
                     str(kwargs.get("stock") or ""),
                     str(kwargs.get("description") or ""),
                     str(kwargs.get("tags") or "")]
            while pipes and not pipes[-1]:
                pipes.pop()
            if pipes:
                tail += " | " + " | ".join(pipes)
            return tail
        if sub == "remove":
            return f"products remove {kwargs.get('product_id')}"
        if sub == "feed":
            return f"products feed {kwargs.get('fmt') or 'json'}"
        return "products list"
    if action == "page":
        return f"page {kwargs.get('product_id')}"
    if action == "pay":
        if sub == "providers":
            return "pay providers"
        if sub == "quote":
            return (f"pay quote {kwargs.get('provider')} "
                    f"{kwargs.get('amount_minor')}")
        if sub == "charge":
            tail = (f"pay charge {kwargs.get('provider')} "
                    f"{kwargs.get('amount_minor')}")
            pipes = [str(kwargs.get("customer") or ""),
                     str(kwargs.get("reference") or ""),
                     str(kwargs.get("channel") or "")]
            while pipes and not pipes[-1]:
                pipes.pop()
            if pipes:
                tail += " | " + " | ".join(pipes)
            return tail
        if sub == "record":
            tail = (f"pay record {kwargs.get('provider')} "
                    f"{kwargs.get('amount_minor')} "
                    f"{kwargs.get('reference')}")
            pipes = [str(kwargs.get("customer") or ""),
                     str(kwargs.get("channel") or "")]
            while pipes and not pipes[-1]:
                pipes.pop()
            if pipes:
                tail += " | " + " | ".join(pipes)
            return tail
        if sub == "list":
            return f"pay list {kwargs.get('provider') or ''}".rstrip()
        if sub == "refund":
            tail = f"pay refund {kwargs.get('reference')}"
            if kwargs.get("amount_minor") is not None:
                tail += f" {kwargs.get('amount_minor')}"
            return tail
        if sub == "dispute":
            return (f"pay dispute {kwargs.get('reference')} "
                    f"{kwargs.get('amount_minor')} {kwargs.get('reason')}")
        if sub == "disputes":
            return "pay disputes"
        if sub == "resolve":
            tail = (f"pay resolve {kwargs.get('dispute_id')} "
                    f"{kwargs.get('outcome')}")
            if kwargs.get("resolution"):
                tail += f" {kwargs.get('resolution')}"
            return tail
        return "pay list"
    if action == "invoice":
        if sub == "create":
            tail = (f"invoice create {kwargs.get('customer')} "
                    f"{kwargs.get('amount_minor')}")
            if kwargs.get("currency"):
                tail += f" | {kwargs.get('currency')}"
            return tail
        if sub == "pay":
            return (f"invoice pay {kwargs.get('invoice_id')} "
                    f"{kwargs.get('payment_reference')}")
        return "invoice list"
    if action == "subscribe":
        if sub == "create":
            tail = (f"subscribe create {kwargs.get('provider')} "
                    f"{kwargs.get('customer')} "
                    f"{kwargs.get('amount_minor')}")
            if kwargs.get("interval"):
                tail += f" | {kwargs.get('interval')}"
            return tail
        if sub == "cancel":
            return f"subscribe cancel {kwargs.get('subscription_id')}"
        return "subscribe list"
    if action == "ship":
        if sub == "quote":
            tail = (f"ship quote {kwargs.get('carrier')} "
                    f"{kwargs.get('weight_kg')}")
            if kwargs.get("distance_km") is not None:
                tail += f" | {kwargs.get('distance_km')}"
            return tail
        if sub == "eta":
            tail = f"ship eta {kwargs.get('carrier')}"
            if kwargs.get("distance_km") is not None:
                tail += f" | {kwargs.get('distance_km')}"
            return tail
        if sub == "send":
            tail = (f"ship send {kwargs.get('carrier')} "
                    f"{kwargs.get('waybill')}")
            pipes = [str(kwargs.get("order_ref") or ""),
                     str(kwargs.get("recipient") or "")]
            while pipes and not pipes[-1]:
                pipes.pop()
            if pipes:
                tail += " | " + " | ".join(pipes)
            return tail
        if sub == "track":
            return (f"ship track {kwargs.get('carrier')} "
                    f"{kwargs.get('waybill')}")
        return "ship list"
    if action == "reconcile":
        return f"reconcile {sub or 'status'}"
    if action == "ticket":
        if sub == "open":
            tail = (f"ticket open {kwargs.get('customer')} "
                    f"{kwargs.get('message')}")
            pipes = [str(kwargs.get("priority") or ""),
                     str(kwargs.get("channel") or ""),
                     str(kwargs.get("phone") or "")]
            while pipes and not pipes[-1]:
                pipes.pop()
            if pipes:
                tail += " | " + " | ".join(pipes)
            return tail
        if sub == "show":
            return f"ticket show {kwargs.get('ticket_id')}"
        if sub == "assign":
            return (f"ticket assign {kwargs.get('ticket_id')} "
                    f"{kwargs.get('assignee')}")
        if sub == "reply":
            tail = f"ticket reply {kwargs.get('ticket_id')}"
            if kwargs.get("template"):
                tail += f" {kwargs.get('template')}"
            return tail
        if sub == "close":
            tail = f"ticket close {kwargs.get('ticket_id')}"
            if kwargs.get("resolution"):
                tail += f" | {kwargs.get('resolution')}"
            return tail
        if sub == "stats":
            return "ticket stats"
        return f"ticket list {kwargs.get('filter') or ''}".rstrip()
    if action == "report":
        return f"report {kwargs.get('month') or ''}".rstrip()
    if action == "tax":
        return (f"tax {kwargs.get('amount_minor')} "
                f"{kwargs.get('jurisdiction') or ''}").rstrip()
    return ""


def register(registry: Any) -> None:
    # Closure over ``registry.context``: ``registry.call`` passes no
    # ``context`` arg, so a ``context``-taking fn would fail with
    # TypeError. Same pattern as ``pulse``/``briefing`` and the other
    # ``__*_`` scheduler bridges in tools/autonomy.py.
    _context = registry.context

    @registry.register(
        "commerce",
        description=(
            "Storefront + cart-recovery operations. "
            "stores: list provisioned stores; store: one store's status; "
            "provision: provision a Medusa/WooCommerce store "
            "(engine=medusa|woocommerce; needs a reachable backend); "
            "recovery_stats / recovery_report: abandoned-cart recovery "
            "performance (recovered revenue); cart_webhook: record a "
            "cart.updated webhook event (opt-in enforced downstream); "
            "run_due_steps: send due recovery messages now. "
            "Commerce organs (same verbs as /store): products (sub=list|"
            "add|remove|feed; product_id, name, price_minor, currency, "
            "category, stock, description, tags, fmt); page (product_id — "
            "cover brief, brief-only); pay (sub=providers|quote|charge|"
            "record|list|refund|dispute|disputes|resolve; provider, "
            "amount_minor, reference, customer, channel, dispute_id, "
            "outcome, resolution, reason); invoice (sub=create|list|pay; "
            "customer, amount_minor, currency, invoice_id, "
            "payment_reference); subscribe (sub=create|list|cancel; "
            "provider, customer, amount_minor, interval, "
            "subscription_id); ship (sub=quote|eta|send|track|list; "
            "carrier, weight_kg, distance_km, waybill, order_ref, "
            "recipient); reconcile (sub=run|status|unreconciled); ticket "
            "(sub=open|list|show|assign|reply|close|stats; customer, "
            "message, priority, channel, phone, ticket_id, assignee, "
            "template, resolution, filter); report (month YYYY-MM); tax "
            "(amount_minor, jurisdiction). Money is integer minor units "
            "(kobo). "
            "action=..."),
        capability="network",
        parameters={
            "action": "str — stores|store|provision|recovery_stats|"
                      "recovery_report|cart_webhook|run_due_steps|"
                      "products|page|pay|invoice|subscribe|ship|"
                      "reconcile|ticket|report|tax",
            "store_id": "str — for store",
            "engine": "str — medusa|woocommerce (provision)",
            "business": "str — business name (provision)",
            "base_url": "str — Medusa backend URL (provision)",
            "site_url": "str — WooCommerce site URL (provision)",
            "admin_email": "str — Medusa admin email (provision)",
            "admin_password": "str — Medusa admin password (provision)",
            "event": "dict — webhook event payload (cart_webhook)",
            "sub": "str — sub-verb for the commerce-organ actions",
            "product_id": "str — products/page",
            "name": "str — products add",
            "price_minor": "int — minor units (kobo)",
            "amount_minor": "int — minor units (kobo)",
            "provider": "str — paystack|flutterwave|monnify|mpesa|stripe",
            "reference": "str — payment reference",
            "customer": "str — customer name",
            "channel": "str — online|whatsapp|in_store",
            "waybill": "str — ship send/track",
            "carrier": "str — gig|kwik|sendbox",
            "month": "str — report YYYY-MM",
            "jurisdiction": "str — tax, e.g. NG",
        },
    )
    def commerce(action: str = "stores", **kwargs: Any) -> dict[str, Any]:
        from ..commerce.medusa import MedusaError

        action = (action or "stores").strip().lower()
        try:
            if action in ("products", "page", "pay", "invoice",
                          "subscribe", "ship", "reconcile", "ticket",
                          "report", "tax"):
                from ..commerce.commands import handle_store_commerce

                home = os.path.expanduser("~/.nomorals")
                reply = handle_store_commerce(
                    _commerce_tail(action, kwargs), home=home)
                return {"ok": True, "reply": reply}
            if action == "stores":
                mgr = _store_manager()
                stores = mgr.list()
                return {"ok": True, "count": len(stores),
                        "stores": [s.to_dict() for s in stores]}
            if action == "store":
                mgr = _store_manager()
                store = mgr.get(str(kwargs.get("store_id") or ""))
                if store is None:
                    return {"ok": False, "error": "store not found"}
                return {"ok": True, "store": store.to_dict()}
            if action == "provision":
                engine = str(kwargs.get("engine") or "medusa").lower()
                business = str(kwargs.get("business") or "")
                mgr = _store_manager()
                try:
                    if engine == "woocommerce":
                        store = mgr.provision_woocommerce(
                            business,
                            site_url=str(kwargs.get("site_url") or ""))
                    else:
                        store = mgr.provision_store(
                            business,
                            base_url=str(kwargs.get("base_url") or
                                         "http://localhost:9000"),
                            admin_email=str(kwargs.get("admin_email") or ""),
                            admin_password=str(
                                kwargs.get("admin_password") or ""))
                except MedusaError as exc:
                    return {"ok": False, "error": str(exc)}
                return {"ok": True, "store": store.to_dict()}
            if action == "recovery_stats":
                rec = _recovery()
                return {"ok": True, **rec.stats()}
            if action == "recovery_report":
                rec = _recovery()
                return {"ok": True, "report": rec.weekly_report()}
            if action == "cart_webhook":
                event = kwargs.get("event") or {}
                if not isinstance(event, dict):
                    return {"ok": False,
                            "error": "event must be a dict"}
                rec = _recovery()
                # handle_webhook normalizes internally (Medusa/Shopify/Woo).
                cart = rec.handle_webhook(event)
                if cart is None:
                    return {"ok": False,
                            "error": "webhook not recognized as a cart event"}
                return {"ok": True, "cart_id": cart.id,
                        "opted_in": cart.opted_in}
            if action == "run_due_steps":
                rec = _recovery()
                sent = rec.run_due_steps()
                return {"ok": True, "messages_sent": sent}
        except Exception as exc:  # noqa: BLE001 - commerce errors are results
            _log.warning("commerce %s failed: %s", action, exc)
            return {"ok": False, "error": str(exc)[:300]}
        return {"ok": False, "error": f"unknown commerce action {action!r}"}

    @registry.register(
        "__dunning_tick__",
        description="Run one commerce dunning tick: charge every due "
        "subscription retry through the real payment gateway, advance "
        "the 1/3/7-day schedule on a failed charge, park the "
        "subscription past_due after the third failure, and journal "
        "every attempt. A missing provider key is a named skip, never "
        "a burned retry. Scheduler-internal; runs daily 09:00. Never "
        "raises — a bad tick is recorded, never fatal to the job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _dunning_tick_tool() -> dict[str, Any]:
        from ..commerce.dunning import run_dunning_tick_once

        try:
            return run_dunning_tick_once(_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            return {"ok": False,
                    "error": f"dunning tick failed: {exc}"}
