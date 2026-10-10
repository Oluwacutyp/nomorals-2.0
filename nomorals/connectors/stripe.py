"""Stripe connector — payments, customers, invoices, payouts.

Drives the Stripe REST API v1 (``https://api.stripe.com/v1``), the most
stable payments API in the industry — these endpoints have been unchanged
for a decade and are used exactly as documented at
https://docs.stripe.com/api:

* ``GET /v1/account`` → account identity (status/test_connection)
* ``GET /v1/balance`` → available + pending balances
* customers: ``POST /v1/customers``, ``GET /v1/customers``,
  ``GET /v1/customers/{id}``
* payment intents: ``POST /v1/payment_intents``,
  ``GET /v1/payment_intents/{id}``,
  ``POST /v1/payment_intents/{id}/confirm|cancel``
* ``POST /v1/payment_links`` → hosted checkout links (line_items[])
* charges: ``GET /v1/charges``, ``GET /v1/charges/{id}``
* invoices: ``POST /v1/invoices``, ``GET /v1/invoices``,
  ``POST /v1/invoices/{id}/finalize|pay|void``
* products/prices: ``POST /v1/products``, ``POST /v1/prices``
* ``POST /v1/transfers`` → move funds to a connected account (gated)
* ``POST /v1/payouts`` → pay out to the bank account (gated)

Auth: secret key (``AuthMethod.API_KEY``) as ``Authorization: Bearer``.
Stripe POSTs are form-encoded (``application/x-www-form-urlencoded``);
nested params use bracket notation (``line_items[0][price]``).

Money safety: ``create_transfer`` and ``create_payout`` move real money
and refuse to run without explicit owner confirmation (``confirmed=True``
after the owner approved the exact payload, or a human checkpoint when
``db`` is given). Invoice finalize/pay likewise. Reads and link/product
creation are not gated.

Auth: API key. Env var ``STRIPE_SECRET_KEY`` (``sk_live_…`` or
``sk_test_…`` — test keys put the connector in test mode, which the
status line reports).
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlencode

from ..core.errors import NoMoralsError, RateLimited
from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["StripeConnector", "StripeError"]

_log = get_logger(__name__)

API_BASE = "https://api.stripe.com/v1"
KEY_ENV = "STRIPE_SECRET_KEY"
DOCS_URL = "https://docs.stripe.com/api"


class StripeError(ConnectorError):
    """A Stripe API call failed."""

    def __init__(
        self,
        message: str,
        *,
        stripe_type: str = "",
        code: str = "",
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.stripe_type = stripe_type
        self.code = code
        self.status_code = status_code


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested dicts/lists → Stripe bracket notation for form encoding."""
    out: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        name = f"{prefix}[{key}]" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                iname = f"{name}[{i}]"
                if isinstance(item, dict):
                    out.update(_flatten(item, iname))
                elif item is not None:
                    out[iname] = str(item)
        elif isinstance(value, bool):
            out[name] = "true" if value else "false"
        else:
            out[name] = str(value)
    return out


@register_connector
class StripeConnector(Connector):
    """Stripe payments on the owner's account."""

    id = "stripe"
    name = "Stripe"
    description = (
        "Stripe payments: balance, customers, payment intents + links, "
        "invoices, transfers and payouts (money moves are gated)"
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── REST ─────────────────────────────────────────────────────────

    def _require_key(self) -> str:
        cred = self._load_credential()
        if cred is None:
            raise StripeError(
                "Stripe is not connected — run "
                "`nm connectors connect --name stripe` "
                "(needs STRIPE_SECRET_KEY from "
                "dashboard.stripe.com → Developers → API keys)"
            )
        return cred.password

    def _api(
        self,
        method: str,
        path: str,
        *,
        key: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        secret = key if key is not None else self._require_key()
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {secret}"}
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            if method == "GET":
                qs = urlencode(_flatten(clean), doseq=True)
                resp = self.http.get(
                    f"{url}?{qs}" if qs else url, headers=headers
                )
            elif method == "POST":
                resp = self.http.post_form(
                    url, _flatten(clean), headers=headers
                )
            elif method == "DELETE":
                resp = self.http.request(
                    "DELETE", url, headers=headers,
                    params=_flatten(clean) or None,
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise StripeError(
                f"Stripe rate limited (429) — back off ~{exc.retry_after:.0f}s",
                stripe_type="rate_limit_error",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise StripeError(f"Stripe request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise StripeError(f"Stripe request failed: {exc}") from exc
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise StripeError(
                f"Stripe {method} {path} returned invalid JSON"
            ) from exc
        err = data.get("error") if isinstance(data, dict) else None
        if err:
            raise StripeError(
                f"Stripe error ({err.get('type', '?')}/"
                f"{err.get('code', '?')}): {err.get('message', '?')}",
                stripe_type=str(err.get("type", "")),
                code=str(err.get("code", "")),
            )
        return data

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self, *, secret_key: str | None = None) -> ConnectResult:
        """Validate a Stripe secret key and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "stripe is already connected — one account per service. "
                "Disconnect first to switch keys."
            )
        key = (secret_key or "").strip() or prompt_secret(
            "Stripe secret key (dashboard → Developers → API keys)",
            env_var=KEY_ENV,
        )
        if not key:
            raise ConnectorError("empty secret key: nothing to connect with")
        if not (key.startswith("sk_live_") or key.startswith("sk_test_")
                or key.startswith("rk_live_")):
            raise ConnectorError(
                "that doesn't look like a Stripe secret key (expected "
                "sk_live_… or sk_test_…)"
            )
        account = self._api("GET", "/v1/account", key=key)
        test_mode = key.startswith("sk_test_")
        label = (
            f"{account.get('business_name') or account.get('email') or 'stripe'}"
            f" ({'test' if test_mode else 'LIVE'} mode)"
        )
        self._store_credential(
            label,
            key,
            credential_type="api_key",
            scopes=["payments:read", "payments:write"],
            metadata={"account_id": account.get("id", ""),
                      "test_mode": test_mode,
                      "country": account.get("country", "")},
        )
        _log.info("stripe connected (%s)", label)
        mode_note = ("TEST MODE — no real money moves."
                     if test_mode else
                     "LIVE MODE — transfers/payouts move real money and "
                     "are confirmation-gated.")
        return ConnectResult(
            ok=True,
            account=label,
            scopes=["payments:read", "payments:write"],
            message=(f"connected to Stripe ({label}). {mode_note} The "
                     "secret key is in the encrypted vault."),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def _mode(self) -> str:
        cred = self._load_credential()
        if cred is None:
            return ""
        return "test" if (cred.metadata or {}).get("test_mode") else "live"

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name stripe` (STRIPE_SECRET_KEY from the Stripe "
                       "dashboard → Developers → API keys)",
            )
        try:
            balance = self._api("GET", "/v1/balance")
        except StripeError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                last_checked=time.time(),
                detail=f"key rejected ({exc}): rotate it in the Stripe "
                       "dashboard and reconnect",
            )
        avail = ", ".join(
            f"{b.get('amount', 0) / 100:,.2f} {b.get('currency', '').upper()}"
            for b in balance.get("available", [])
        ) or "0"
        return ConnectorStatus(
            connected=True,
            account=cred.username,
            scopes=["payments:read", "payments:write"],
            last_checked=time.time(),
            detail=f"available={avail} ({self._mode()} mode)",
        )

    def test_connection(self) -> bool:
        try:
            return self.status().connected
        except Exception:  # noqa: BLE001 - offline / bad key → not linked
            return False

    # ── reads ────────────────────────────────────────────────────────

    def get_balance(self) -> dict[str, Any]:
        """Available + pending balances."""
        return self._api("GET", "/v1/balance")

    def list_customers(
        self, *, limit: int = 20, email: str = ""
    ) -> list[dict[str, Any]]:
        """Customers, newest first; filter by exact email."""
        data = self._api(
            "GET", "/v1/customers",
            params={"limit": max(1, min(100, limit)),
                    "email": email or None},
        )
        return list(data.get("data") or [])

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        customer_id = (customer_id or "").strip()
        if not customer_id:
            raise StripeError("empty customer_id")
        return self._api("GET", f"/v1/customers/{customer_id}")

    def create_customer(
        self, *, name: str = "", email: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        """Create a customer record (no money moves)."""
        if not name and not email:
            raise StripeError("create_customer needs at least a name or email")
        return self._api(
            "POST", "/v1/customers",
            params={"name": name or None, "email": email or None, **extra},
        )

    def list_payment_intents(
        self, *, limit: int = 20, customer: str = ""
    ) -> list[dict[str, Any]]:
        data = self._api(
            "GET", "/v1/payment_intents",
            params={"limit": max(1, min(100, limit)),
                    "customer": customer or None},
        )
        return list(data.get("data") or [])

    def get_payment_intent(self, intent_id: str) -> dict[str, Any]:
        intent_id = (intent_id or "").strip()
        if not intent_id:
            raise StripeError("empty payment intent id")
        return self._api("GET", f"/v1/payment_intents/{intent_id}")

    def create_payment_intent(
        self,
        amount: int,
        currency: str,
        *,
        customer: str = "",
        description: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        """Create a payment intent (amount in the SMALLEST unit — kobo,
        cents). Nothing is charged until it is confirmed."""
        if amount <= 0:
            raise StripeError("amount must be positive (smallest unit)")
        currency = (currency or "").strip().lower()
        if not currency:
            raise StripeError("currency is required (e.g. 'usd', 'ngn')")
        return self._api(
            "POST", "/v1/payment_intents",
            params={"amount": amount, "currency": currency,
                    "customer": customer or None,
                    "description": description or None, **extra},
        )

    def create_payment_link(
        self, line_items: list[dict[str, Any]], **extra: Any
    ) -> dict[str, Any]:
        """Hosted checkout link. ``line_items`` like
        ``[{"price": "price_…", "quantity": 1}]`` or ad-hoc
        ``[{"price_data": {"currency": "usd", "unit_amount": 500,
        "product_data": {"name": "…"}}, "quantity": 1}]``."""
        if not line_items:
            raise StripeError("create_payment_link needs line_items")
        return self._api(
            "POST", "/v1/payment_links",
            params={"line_items": line_items, **extra},
        )

    def list_charges(
        self, *, limit: int = 20, customer: str = ""
    ) -> list[dict[str, Any]]:
        data = self._api(
            "GET", "/v1/charges",
            params={"limit": max(1, min(100, limit)),
                    "customer": customer or None},
        )
        return list(data.get("data") or [])

    def list_invoices(
        self, *, limit: int = 20, customer: str = "", status: str = ""
    ) -> list[dict[str, Any]]:
        data = self._api(
            "GET", "/v1/invoices",
            params={"limit": max(1, min(100, limit)),
                    "customer": customer or None,
                    "status": status or None},
        )
        return list(data.get("data") or [])

    def create_invoice(
        self, customer: str, *, auto_advance: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Draft invoice for a customer (add items first via invoice
        items, then finalize)."""
        customer = (customer or "").strip()
        if not customer:
            raise StripeError("create_invoice needs a customer id")
        return self._api(
            "POST", "/v1/invoices",
            params={"customer": customer, "auto_advance": auto_advance,
                    **extra},
        )

    def create_product(self, name: str, **extra: Any) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise StripeError("create_product needs a name")
        return self._api("POST", "/v1/products",
                         params={"name": name, **extra})

    def create_price(
        self, product: str, unit_amount: int, currency: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Price for a product (amount in smallest unit)."""
        if unit_amount <= 0:
            raise StripeError("unit_amount must be positive (smallest unit)")
        return self._api(
            "POST", "/v1/prices",
            params={"product": (product or '').strip(),
                    "unit_amount": unit_amount,
                    "currency": (currency or '').strip().lower(), **extra},
        )

    # ── money-moving calls (confirmation-gated) ──────────────────────

    def _gate_money(
        self,
        *,
        confirmed: bool,
        db: Any,
        context: Any,
        stage: str,
        title: str,
        instructions: str,
        resume_state: dict[str, Any],
    ) -> None:
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage=stage,
            title=title,
            instructions=instructions,
            resume_state=resume_state,
        )

    def confirm_payment_intent(
        self,
        intent_id: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Confirm a payment intent — CHARGES the customer. Gated."""
        intent_id = (intent_id or "").strip()
        if not intent_id:
            raise StripeError("empty payment intent id")
        intent = self.get_payment_intent(intent_id)
        amount = intent.get("amount", 0) / 100
        currency = str(intent.get("currency", "")).upper()
        self._gate_money(
            confirmed=confirmed, db=db, context=context,
            stage="confirm_payment_intent",
            title=f"Stripe: charge {amount:,.2f} {currency}",
            instructions="\n".join([
                "Devon wants to CONFIRM a Stripe payment intent — this "
                "CHARGES the customer real money.",
                f"intent={intent_id} amount={amount:,.2f} {currency} "
                f"customer={intent.get('customer', '—')}",
                f"description={intent.get('description', '—')}",
                "Approve ONLY if this charge is exactly what you want.",
            ]),
            resume_state={"op": "confirm_payment_intent",
                          "intent_id": intent_id},
        )
        return self._api(
            "POST", f"/v1/payment_intents/{intent_id}/confirm")

    def finalize_invoice(
        self,
        invoice_id: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Finalize a draft invoice — locks it and (with collection
        method charge_automatically) attempts payment. Gated."""
        invoice_id = (invoice_id or "").strip()
        if not invoice_id:
            raise StripeError("empty invoice id")
        self._gate_money(
            confirmed=confirmed, db=db, context=context,
            stage="finalize_invoice",
            title=f"Stripe: finalize invoice {invoice_id}",
            instructions="\n".join([
                "Devon wants to FINALIZE a Stripe invoice — this locks the "
                "invoice and may charge the customer.",
                f"invoice={invoice_id}",
                "Approve ONLY if the invoice is exactly what you want sent.",
            ]),
            resume_state={"op": "finalize_invoice", "invoice_id": invoice_id},
        )
        return self._api("POST", f"/v1/invoices/{invoice_id}/finalize")

    def create_transfer(
        self,
        amount: int,
        currency: str,
        destination: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Move funds to a connected Stripe account. Gated."""
        if amount <= 0:
            raise StripeError("amount must be positive (smallest unit)")
        destination = (destination or "").strip()
        if not destination:
            raise StripeError("destination account id is required")
        currency = (currency or "").strip().lower()
        self._gate_money(
            confirmed=confirmed, db=db, context=context,
            stage="create_transfer",
            title=f"Stripe: transfer {amount / 100:,.2f} "
                  f"{currency.upper()} → {destination}",
            instructions="\n".join([
                "Devon wants to TRANSFER real money out of your Stripe "
                "balance to a connected account.",
                f"amount={amount / 100:,.2f} {currency.upper()} "
                f"destination={destination}",
                "Approve ONLY if the amount and destination are exactly "
                "what you want.",
            ]),
            resume_state={"op": "create_transfer", "amount": amount,
                          "currency": currency, "destination": destination},
        )
        return self._api(
            "POST", "/v1/transfers",
            params={"amount": amount, "currency": currency,
                    "destination": destination},
        )

    def create_payout(
        self,
        amount: int,
        currency: str,
        *,
        destination: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Pay out from the Stripe balance to a bank account. Gated."""
        if amount <= 0:
            raise StripeError("amount must be positive (smallest unit)")
        currency = (currency or "").strip().lower()
        self._gate_money(
            confirmed=confirmed, db=db, context=context,
            stage="create_payout",
            title=f"Stripe: payout {amount / 100:,.2f} {currency.upper()}",
            instructions="\n".join([
                "Devon wants to PAY OUT real money from your Stripe "
                "balance to your bank account.",
                f"amount={amount / 100:,.2f} {currency.upper()} "
                f"destination={destination or 'default bank account'}",
                "Approve ONLY if this payout is exactly what you want.",
            ]),
            resume_state={"op": "create_payout", "amount": amount,
                          "currency": currency, "destination": destination},
        )
        return self._api(
            "POST", "/v1/payouts",
            params={"amount": amount, "currency": currency,
                    "destination": destination or None},
        )

    # ── checkpoint resume ────────────────────────────────────────────

    def resume_checkpoint(self, checkpoint: Any, *, db: Any,
                          context: Any = None) -> dict[str, Any]:
        """Complete a gated money move after its checkpoint resolved."""
        state = dict(getattr(checkpoint, "resume_state", None) or {})
        op = state.get("op", "")
        if state.get("stage") not in {
            "confirm_payment_intent", "finalize_invoice", "create_transfer",
            "create_payout",
        }:
            raise StripeError(
                f"cannot resume checkpoint stage {state.get('stage')!r}"
            )
        if op == "confirm_payment_intent":
            return self.confirm_payment_intent(
                state["intent_id"], confirmed=True, db=db, context=context)
        if op == "finalize_invoice":
            return self.finalize_invoice(
                state["invoice_id"], confirmed=True, db=db, context=context)
        if op == "create_transfer":
            return self.create_transfer(
                int(state["amount"]), state["currency"], state["destination"],
                confirmed=True, db=db, context=context)
        if op == "create_payout":
            return self.create_payout(
                int(state["amount"]), state["currency"],
                destination=state.get("destination", ""),
                confirmed=True, db=db, context=context)
        raise StripeError(f"unknown checkpoint op {op!r}")
