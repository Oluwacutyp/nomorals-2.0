"""Commerce tool — storefronts + cart recovery as an agent surface.

The commerce package (``nomorals/commerce/``) had real implementations
(Medusa/WooCommerce provisioning, WhatsApp cart recovery) but almost no
reachable surface: provisioning was only callable from one odd corner of
the partner runtime, and cart recovery had no chat/CLI path at all.
This module is that surface.

Actions: stores | store | provision | recovery_stats | recovery_report |
         cart_webhook | run_due_steps

Read actions are always safe. ``provision`` creates real infrastructure
(needs a reachable Medusa backend or WooCommerce site) and fails closed
with the reason when the backend isn't there. ``cart_webhook`` only
*records* carts — the 3-step WhatsApp sequence still requires the
customer's explicit opt-in, enforced inside CartRecovery.
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


def register(registry: Any) -> None:
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
            "action=..."),
        capability="network",
        parameters={
            "action": "str — stores|store|provision|recovery_stats|"
                      "recovery_report|cart_webhook|run_due_steps",
            "store_id": "str — for store",
            "engine": "str — medusa|woocommerce (provision)",
            "business": "str — business name (provision)",
            "base_url": "str — Medusa backend URL (provision)",
            "site_url": "str — WooCommerce site URL (provision)",
            "admin_email": "str — Medusa admin email (provision)",
            "admin_password": "str — Medusa admin password (provision)",
            "event": "dict — webhook event payload (cart_webhook)",
        },
    )
    def commerce(action: str = "stores", **kwargs: Any) -> dict[str, Any]:
        from ..commerce.medusa import MedusaError

        action = (action or "stores").strip().lower()
        try:
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
