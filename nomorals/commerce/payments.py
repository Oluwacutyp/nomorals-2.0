"""Payments: local-first rails — Paystack, Flutterwave, Monnify, M-Pesa + Stripe.

Local rails first: Nigerian / African businesses get paid through the
rails their customers actually use (Paystack, Flutterwave, Monnify,
M-Pesa) before the international fallback (Stripe).

Hard rule — never fake success:
- Every provider requires its real secret key (explicit ``keys`` dict or
  the documented env var). Missing key → :class:`PaymentError`, never a
  fabricated "success".
- A real charge needs a real HTTP round-trip. ``charge()`` goes through an
  injectable ``http`` client (``http(method, url, headers, json_body)``).
  With no HTTP client configured, ``charge()`` raises — it does NOT
  pretend the money moved. Tests inject a fake HTTP client; production
  wires a real one (urllib/requests).
- Webhook verification is HMAC over the raw body with the provider
  secret, compared with :func:`hmac.compare_digest`.

Money is integer minor units everywhere (kobo / cents / pesewas),
matching :mod:`nomorals.finance.ledger`. Fees are computed from the
provider descriptor's published schedule — the descriptor says when a
number is a published rate vs an estimate, and operators should re-check
provider docs before quoting fees to a customer.

Fee schedules (published at time of writing, verify before quoting):
- Paystack NG: 1.5% + ₦100, capped at ₦2,000 (local cards).
- Flutterwave NG: 1.4%, capped at ₦2,000 (local cards).
- Monnify NG: 1.0%, capped at ₦1,000.
- M-Pesa KE: varies by transaction band — modelled as 1.0% estimate.
- Stripe: 2.9% + 30¢ (international cards); Stripe is NOT a Nigerian
  local rail — it is the USD/international fallback.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PaymentError",
    "Provider",
    "PROVIDERS",
    "provider_for",
    "Payment",
    "Invoice",
    "Subscription",
    "Dispute",
    "PaymentGateway",
]


class PaymentError(RuntimeError):
    """A payment operation could not be completed for real."""


@dataclass(frozen=True)
class Provider:
    """Descriptor for one payment rail."""
    name: str
    country: str          # primary market, ISO code
    currency: str         # default settlement currency, ISO code
    fee_pct_bps: int      # fee percent in basis points (150 = 1.5%)
    fee_fixed_minor: int  # fixed fee in minor units (100 = ₦1 / $1... see below)
    fee_cap_minor: int | None  # None = uncapped
    fee_note: str
    env_key: str          # env var holding the secret key
    supports_subscriptions: bool = True
    supports_refunds: bool = True

    def fee_for(self, amount_minor: int) -> int:
        """Fee in minor units for ``amount_minor``. Integer math only."""
        if amount_minor <= 0:
            raise PaymentError("amount must be positive")
        fee = (amount_minor * self.fee_pct_bps) // 10_000 + self.fee_fixed_minor
        if self.fee_cap_minor is not None:
            fee = min(fee, self.fee_cap_minor)
        return fee

    def net_for(self, amount_minor: int) -> int:
        return amount_minor - self.fee_for(amount_minor)


PROVIDERS: dict[str, Provider] = {
    "paystack": Provider(
        name="Paystack", country="NG", currency="NGN",
        fee_pct_bps=150, fee_fixed_minor=10_000, fee_cap_minor=200_000,
        fee_note="1.5% + ₦100, capped ₦2,000 (local cards, published).",
        env_key="PAYSTACK_SECRET_KEY",
    ),
    "flutterwave": Provider(
        name="Flutterwave", country="NG", currency="NGN",
        fee_pct_bps=140, fee_fixed_minor=0, fee_cap_minor=200_000,
        fee_note="1.4%, capped ₦2,000 (local cards, published).",
        env_key="FLUTTERWAVE_SECRET_KEY",
    ),
    "monnify": Provider(
        name="Monnify", country="NG", currency="NGN",
        fee_pct_bps=100, fee_fixed_minor=0, fee_cap_minor=100_000,
        fee_note="1.0%, capped ₦1,000 (published).",
        env_key="MONNIFY_SECRET_KEY",
    ),
    "mpesa": Provider(
        name="M-Pesa", country="KE", currency="KES",
        fee_pct_bps=100, fee_fixed_minor=0, fee_cap_minor=None,
        fee_note="Band-based in reality; 1.0% here is an ESTIMATE. "
                 "Confirm Daraja tariff before quoting.",
        env_key="MPESA_CONSUMER_SECRET",
        supports_subscriptions=False,
    ),
    "stripe": Provider(
        name="Stripe", country="US", currency="USD",
        fee_pct_bps=290, fee_fixed_minor=30, fee_cap_minor=None,
        fee_note="2.9% + 30¢ international. NOT a Nigerian local rail — "
                 "USD/international fallback only.",
        env_key="STRIPE_SECRET_KEY",
    ),
}


def provider_for(name: str) -> Provider:
    """Look up a provider descriptor by name (case-insensitive)."""
    key = (name or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {"mpesa": "mpesa", "m-pesa": "mpesa", "m_pesa": "mpesa"}
    key = aliases.get(key, key)
    if key not in PROVIDERS:
        raise PaymentError(
            f"unknown provider {name!r} — known: {sorted(PROVIDERS)}")
    return PROVIDERS[key]


# ── records ───────────────────────────────────────────────────────────────

@dataclass
class Payment:
    id: str
    provider: str
    amount_minor: int
    currency: str
    reference: str
    status: str  # succeeded | failed | refunded | pending
    customer: str = ""
    channel: str = "online"  # online | whatsapp | in_store | ...
    fee_minor: int = 0
    created_at: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def net_minor(self) -> int:
        return self.amount_minor - self.fee_minor

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "provider": self.provider,
            "amount_minor": self.amount_minor, "currency": self.currency,
            "reference": self.reference, "status": self.status,
            "customer": self.customer, "channel": self.channel,
            "fee_minor": self.fee_minor, "created_at": self.created_at,
        }


@dataclass
class Invoice:
    id: str
    customer: str
    amount_minor: int
    currency: str = "NGN"
    status: str = "unpaid"  # unpaid | paid | overdue | void
    payment_reference: str = ""
    created_at: float = 0.0
    paid_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "customer": self.customer,
            "amount_minor": self.amount_minor, "currency": self.currency,
            "status": self.status,
            "payment_reference": self.payment_reference,
            "created_at": self.created_at, "paid_at": self.paid_at,
        }


@dataclass
class Subscription:
    id: str
    provider: str
    customer: str
    amount_minor: int
    currency: str = "NGN"
    interval: str = "monthly"  # daily | weekly | monthly | yearly
    status: str = "active"  # active | past_due | cancelled
    created_at: float = 0.0
    cancelled_at: float = 0.0
    failed_attempts: int = 0
    next_retry_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "provider": self.provider,
            "customer": self.customer, "amount_minor": self.amount_minor,
            "currency": self.currency, "interval": self.interval,
            "status": self.status, "created_at": self.created_at,
            "cancelled_at": self.cancelled_at,
            "failed_attempts": self.failed_attempts,
            "next_retry_at": self.next_retry_at,
        }


@dataclass
class Dispute:
    id: str
    payment_reference: str
    amount_minor: int
    reason: str
    status: str = "open"  # open | won | lost | accepted
    resolution: str = ""
    created_at: float = 0.0
    resolved_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "payment_reference": self.payment_reference,
            "amount_minor": self.amount_minor, "reason": self.reason,
            "status": self.status, "resolution": self.resolution,
            "created_at": self.created_at, "resolved_at": self.resolved_at,
        }


# Dunning: retry a failed subscription charge after 1, 3, then 7 days.
# After the third failed retry the subscription goes past_due and stops
# retrying automatically — a human (or the owner) decides next.
DUNNING_SCHEDULE_DAYS: tuple[int, ...] = (1, 3, 7)

#: Provider charge endpoints (real paths; only used when http is wired).
_CHARGE_URLS = {
    "paystack": "https://api.paystack.co/transaction/initialize",
    "flutterwave": "https://api.flutterwave.com/v3/payments",
    "monnify": "https://api.monnify.com/api/v1/merchant/transactions/init-transaction",
    "mpesa": "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest",
    "stripe": "https://api.stripe.com/v1/payment_intents",
}


class PaymentGateway:
    """Payment operations across providers. In-memory record store.

    Args:
        keys: explicit ``{provider_name: secret_key}``. Wins over env vars.
        http: ``http(method, url, headers, json_body) -> (status, payload)``.
            None means NO real charging is possible — ``charge()`` raises
            instead of faking success. Tests inject a fake; production
            wires urllib/requests.
        now: injectable clock for tests.
    """

    def __init__(self, *, keys: dict[str, str] | None = None,
                 http: Callable[..., Any] | None = None,
                 now: Callable[[], float] | None = None) -> None:
        self._keys = dict(keys or {})
        self._http = http
        self._now = now or time.time
        self.payments: list[Payment] = []
        self.invoices: list[Invoice] = []
        self.subscriptions: list[Subscription] = []
        self.disputes: list[Dispute] = []

    # ── keys ──

    def _secret(self, provider: Provider) -> str:
        normalized = "".join(
            c for c in provider.name.lower() if c.isalnum())
        for key in (provider.name.lower(), provider.name, normalized):
            value = self._keys.get(key)
            if value:
                return str(value).strip()
        return str(os.environ.get(provider.env_key, "") or "").strip()

    def require_configured(self, name: str) -> Provider:
        """Return the provider descriptor or raise when its key is missing."""
        provider = provider_for(name)
        if not self._secret(provider):
            raise PaymentError(
                f"{provider.name} secret key missing — set {provider.env_key} "
                f"or pass keys={{'{provider.name.lower()}': '...'}}. "
                "Refusing to fake a charge.")
        return provider

    # ── charges / refunds ──

    def charge(self, provider_name: str, amount_minor: int, *,
               currency: str = "", customer: str = "",
               reference: str = "", channel: str = "online",
               metadata: dict[str, Any] | None = None) -> Payment:
        """Charge a customer through ``provider_name`` — for real.

        Raises PaymentError when the provider key is missing, when no HTTP
        client is configured, when the gateway answers non-2xx, or when the
        amount is invalid. Never returns a fabricated success.
        """
        provider = self.require_configured(provider_name)
        if amount_minor <= 0:
            raise PaymentError("amount_minor must be positive")
        if self._http is None:
            raise PaymentError(
                f"no HTTP client configured — cannot charge ₦/real money "
                f"through {provider.name} without a real gateway round-trip. "
                "Wire http=... to enable live charges.")
        currency = (currency or provider.currency).upper()
        ref = reference or f"devon_{uuid.uuid4().hex[:12]}"
        url = _CHARGE_URLS[provider.name.lower()]
        try:
            status, payload = self._http(
                "POST", url,
                {"Authorization": f"Bearer {self._secret(provider)}"},
                {"amount": amount_minor, "currency": currency,
                 "reference": ref, "customer": customer,
                 "metadata": metadata or {}},
            )
        except Exception as exc:  # noqa: BLE001 — gateway failure is a failure
            raise PaymentError(
                f"{provider.name} charge failed (transport): {exc}") from exc
        if not (200 <= int(status) < 300):
            raise PaymentError(
                f"{provider.name} charge rejected: HTTP {status}: "
                f"{str(payload)[:200]}")
        fee = provider.fee_for(amount_minor)
        payment = Payment(
            id="pay_" + uuid.uuid4().hex[:12], provider=provider.name.lower(),
            amount_minor=int(amount_minor), currency=currency,
            reference=ref, status="succeeded", customer=customer,
            channel=channel, fee_minor=fee, created_at=self._now(),
            raw=payload if isinstance(payload, dict) else {"raw": payload},
        )
        self.payments.append(payment)
        _log.info("charged %s %s via %s (ref %s)",
                  amount_minor, currency, provider.name, ref)
        return payment

    def refund(self, payment: Payment, *, amount_minor: int | None = None,
               reason: str = "") -> Payment:
        """Refund a succeeded payment. Requires the provider key + HTTP.

        The refunded amount is validated against the original payment;
        the original record is marked ``refunded`` (full) or kept with a
        refund record appended for partial refunds.
        """
        provider = self.require_configured(payment.provider)
        if payment.status not in ("succeeded",):
            raise PaymentError(
                f"cannot refund a {payment.status} payment (ref {payment.reference})")
        amount = payment.amount_minor if amount_minor is None else int(amount_minor)
        if amount <= 0 or amount > payment.amount_minor:
            raise PaymentError(
                f"refund amount {amount} out of range for payment "
                f"{payment.amount_minor}")
        if self._http is None:
            raise PaymentError(
                "no HTTP client configured — cannot refund for real.")
        try:
            status, payload = self._http(
                "POST", _CHARGE_URLS[provider.name.lower()] + "/refund",
                {"Authorization": f"Bearer {self._secret(provider)}"},
                {"reference": payment.reference, "amount": amount,
                 "reason": reason},
            )
        except Exception as exc:  # noqa: BLE001
            raise PaymentError(f"{provider.name} refund failed: {exc}") from exc
        if not (200 <= int(status) < 300):
            raise PaymentError(
                f"{provider.name} refund rejected: HTTP {status}")
        refund = Payment(
            id="ref_" + uuid.uuid4().hex[:12], provider=payment.provider,
            amount_minor=-amount, currency=payment.currency,
            reference=payment.reference + "-refund",
            status="refunded", customer=payment.customer,
            channel=payment.channel, fee_minor=0, created_at=self._now(),
            raw={"refunded_payment": payment.id, "reason": reason},
        )
        self.payments.append(refund)
        if amount == payment.amount_minor:
            payment.status = "refunded"
        return refund

    # ── webhooks ──

    def webhook_verify(self, provider_name: str, raw_body: bytes,
                       signature: str) -> bool:
        """Verify a provider webhook signature (HMAC-SHA256, constant-time).

        Paystack sends ``x-paystack-signature``; Flutterwave sends the
        secret itself as ``verif-hash``; Monnify/M-Pesa/Stripe verifications
        reduce to the same HMAC check here with the provider secret.
        Returns False (never raises) on missing key / bad signature.
        """
        try:
            provider = provider_for(provider_name)
            secret = self._secret(provider)
            if not secret or not signature:
                return False
            expected = hmac.new(
                secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, signature.strip())
        except Exception:  # noqa: BLE001 — verification failure is False
            return False

    # ── subscriptions + dunning ──

    def create_subscription(self, provider_name: str, customer: str,
                            amount_minor: int, *,
                            currency: str = "",
                            interval: str = "monthly") -> Subscription:
        provider = self.require_configured(provider_name)
        if not provider.supports_subscriptions:
            raise PaymentError(
                f"{provider.name} does not support subscriptions via this "
                "module (M-Pesa: use STK-push per payment instead).")
        if amount_minor <= 0:
            raise PaymentError("amount_minor must be positive")
        if interval not in ("daily", "weekly", "monthly", "yearly"):
            raise PaymentError(f"unknown interval {interval!r}")
        sub = Subscription(
            id="sub_" + uuid.uuid4().hex[:12], provider=provider.name.lower(),
            customer=customer, amount_minor=int(amount_minor),
            currency=(currency or provider.currency).upper(),
            interval=interval, created_at=self._now(),
        )
        self.subscriptions.append(sub)
        return sub

    def cancel_subscription(self, sub: Subscription) -> Subscription:
        if sub.status == "cancelled":
            return sub
        sub.status = "cancelled"
        sub.cancelled_at = self._now()
        return sub

    def record_failed_charge(self, sub: Subscription) -> Subscription:
        """Dunning: schedule the next retry, or go past_due when exhausted.

        Retry 1 → +1 day, retry 2 → +3 days, retry 3 → +7 days. After the
        third failed retry the subscription is ``past_due`` and stops
        auto-retrying.
        """
        sub.failed_attempts += 1
        idx = sub.failed_attempts - 1
        if idx < len(DUNNING_SCHEDULE_DAYS):
            sub.next_retry_at = (
                self._now() + DUNNING_SCHEDULE_DAYS[idx] * 86400)
            sub.status = "active"
        else:
            sub.next_retry_at = 0.0
            sub.status = "past_due"
        return sub

    def dunning_due(self) -> list[Subscription]:
        """Subscriptions whose retry time has arrived and are still active."""
        now = self._now()
        return [s for s in self.subscriptions
                if s.status == "active" and s.failed_attempts
                and 0 < s.next_retry_at <= now]

    # ── invoices ──

    def create_invoice(self, customer: str, amount_minor: int, *,
                       currency: str = "NGN") -> Invoice:
        if amount_minor <= 0:
            raise PaymentError("invoice amount must be positive")
        inv = Invoice(
            id="inv_" + uuid.uuid4().hex[:12], customer=customer,
            amount_minor=int(amount_minor), currency=currency.upper(),
            created_at=self._now(),
        )
        self.invoices.append(inv)
        return inv

    def mark_paid(self, invoice: Invoice, payment: Payment) -> Invoice:
        """Mark an invoice paid by a real payment. Amounts must match."""
        if payment.status != "succeeded":
            raise PaymentError(
                f"cannot mark invoice paid with a {payment.status} payment")
        if payment.amount_minor != invoice.amount_minor:
            raise PaymentError(
                f"payment {payment.amount_minor} does not match invoice "
                f"{invoice.amount_minor} — refusing to mark paid")
        invoice.status = "paid"
        invoice.payment_reference = payment.reference
        invoice.paid_at = self._now()
        return invoice

    # ── disputes ──

    def record_dispute(self, payment_reference: str, amount_minor: int,
                       reason: str) -> Dispute:
        if amount_minor <= 0:
            raise PaymentError("dispute amount must be positive")
        dispute = Dispute(
            id="dsp_" + uuid.uuid4().hex[:12],
            payment_reference=payment_reference,
            amount_minor=int(amount_minor), reason=reason,
            created_at=self._now(),
        )
        self.disputes.append(dispute)
        return dispute

    def resolve_dispute(self, dispute: Dispute, outcome: str,
                        resolution: str = "") -> Dispute:
        """outcome: won | lost | accepted. Anything else raises."""
        if outcome not in ("won", "lost", "accepted"):
            raise PaymentError(
                f"unknown dispute outcome {outcome!r} — won|lost|accepted")
        if dispute.status != "open":
            raise PaymentError(f"dispute {dispute.id} already {dispute.status}")
        dispute.status = outcome
        dispute.resolution = resolution
        dispute.resolved_at = self._now()
        return dispute

    # ── lookups ──

    def find_payment(self, reference: str) -> Payment | None:
        return next((p for p in self.payments if p.reference == reference),
                    None)

    def to_json(self) -> str:
        return json.dumps({
            "payments": [p.to_dict() for p in self.payments],
            "invoices": [i.to_dict() for i in self.invoices],
            "subscriptions": [s.to_dict() for s in self.subscriptions],
            "disputes": [d.to_dict() for d in self.disputes],
        }, indent=2)
