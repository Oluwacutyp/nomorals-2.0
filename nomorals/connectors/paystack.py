"""Paystack connector — Nigerian payments.

Docs: https://paystack.com/docs/api/

Auth: one secret key (dashboard: https://dashboard.paystack.com ->
Settings -> API Keys & Webhooks; ``sk_test_*`` for test mode,
``sk_live_*`` for production) in the ``Authorization: Bearer`` header,
``AuthMethod.API_KEY``. Amounts are in minor units (kobo for NGN).

Money movement is confirmation-gated: ``initialize_transaction`` only
creates a checkout URL (no money moves until the payer acts), while
``charge_authorization`` pulls money off a saved card and refuses to run
without explicit owner confirmation (``confirmed=True`` after the owner
approved the exact charge, or a human checkpoint when ``db`` is given).

Webhook handling: Paystack notifies ``charge.success`` etc. on the URL
configured in the dashboard. ``verify_webhook_signature`` checks the
``x-paystack-signature`` HMAC-SHA512 so a local webhook receiver can trust
the payload — receiving the webhook itself is the deployer's web server,
not this connector.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

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
from .checkpoints import CheckpointState
from .registry import register_connector

__all__ = ["PaystackConnector", "PaystackError"]

_log = get_logger(__name__)

API_BASE = "https://api.paystack.co"
SECRET_ENV = "PAYSTACK_SECRET_KEY"
DOCS_URL = "https://paystack.com/docs/api"


class PaystackError(ConnectorError):
    """A Paystack API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class PaystackConnector(Connector):
    """Devon's Paystack adapter: collect and verify payments."""

    id = "paystack"
    name = "Paystack"
    description = (
        "Nigerian payments: initialize transactions (checkout URLs), "
        "verify them, list transaction history, manage customers, and "
        "charge saved authorizations (confirmation-gated). Authenticates "
        "with a Paystack secret key (Bearer header)."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        secret_key: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a secret key and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "paystack is already connected — one account per service. "
                "Disconnect first to switch secret keys."
            )
        secret = (secret_key or "").strip() or prompt_secret(
            "Paystack secret key", env_var=SECRET_ENV
        )
        if not secret:
            raise ConnectorError("empty secret key: nothing to connect with")
        mode = "live" if secret.startswith("sk_live_") else "test"
        self._validate_key(secret)
        self._store_credential(
            "paystack",
            secret,
            credential_type="api_key",
            scopes=["transactions", "customers"],
            metadata={"mode": mode},
        )
        _log.info("paystack connected (%s mode)", mode)
        return ConnectResult(
            ok=True,
            account=f"paystack ({mode})",
            scopes=["transactions", "customers"],
            message=(
                f"connected to Paystack in {mode} mode. The secret key is "
                "in the encrypted vault. initialize_transaction() creates "
                "checkout URLs; charge_authorization() needs explicit "
                "owner confirmation every time."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name paystack`",
            )
        try:
            self._api("GET", "/transaction", params={"perPage": 1},
                      secret=cred.password)
        except PaystackError as exc:
            return ConnectorStatus(
                connected=False,
                account="paystack",
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"secret key rejected ({exc}): reconnect with a "
                       "fresh key from the Paystack dashboard",
            )
        mode = (cred.metadata or {}).get("mode", "?")
        return ConnectorStatus(
            connected=True,
            account=f"paystack ({mode})",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="secret key valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/transaction", params={"perPage": 1},
                      secret=cred.password)
            return True
        except ConnectorError:
            return False

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Complete a confirmed charge after the owner resolved it."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve the charge first"
            )
        if (checkpoint.resume_state or {}).get("stage") != "charge":
            raise ConnectorError(
                "paystack cannot resume checkpoint stage "
                f"{(checkpoint.resume_state or {}).get('stage')!r}"
            )
        payload = dict((checkpoint.resume_state or {}).get("payload", {}))
        if not payload.get("authorization_code") or not payload.get(
            "amount_kobo"
        ):
            raise ConnectorError(
                "the resolved checkpoint has no charge payload — "
                "it cannot charge"
            )
        return self._charge_now(payload)

    # ── transactions ─────────────────────────────────────────────

    def initialize_transaction(
        self,
        email: str,
        amount_kobo: int,
        *,
        callback_url: str = "",
        reference: str = "",
        currency: str = "NGN",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a checkout session (``POST /transaction/initialize``).

        ``amount_kobo`` is in minor units (500000 = ₦5,000.00). Returns
        ``authorization_url`` (hand it to the payer), ``access_code`` and
        ``reference``. No money moves here — the payer completes checkout
        themselves; confirm with :meth:`verify_transaction`.
        """
        email = (email or "").strip()
        if not email or "@" not in email:
            raise ConnectorError(
                f"invalid customer email {email!r}: a transaction needs a "
                "real email address"
            )
        if amount_kobo <= 0:
            raise ConnectorError(
                f"invalid amount {amount_kobo}: must be a positive integer "
                "in kobo"
            )
        payload: dict[str, Any] = {
            "email": email,
            "amount": amount_kobo,
            "currency": currency.upper(),
        }
        if callback_url:
            payload["callback_url"] = callback_url
        if reference:
            payload["reference"] = reference
        if metadata:
            payload["metadata"] = metadata
        data = self._api("POST", "/transaction/initialize", payload=payload)
        _log.info(
            "paystack transaction initialized: %s (%d %s)",
            data.get("reference", "?"), amount_kobo, currency.upper(),
        )
        return data

    def verify_transaction(self, reference: str) -> dict[str, Any]:
        """Confirm a payment's status (``GET /transaction/verify/:ref``).

        The source of truth after checkout: ``data.status`` is ``success``,
        ``failed``, or ``abandoned``. Never trust a redirect alone.
        """
        reference = (reference or "").strip()
        if not reference:
            raise ConnectorError("empty transaction reference")
        data = self._api("GET", f"/transaction/verify/{reference}")
        return {
            "reference": reference,
            "status": data.get("status", ""),
            "amount_kobo": data.get("amount", 0),
            "currency": data.get("currency", ""),
            "paid_at": data.get("paid_at", ""),
            "customer": data.get("customer", {}),
            "raw": data,
        }

    def get_transaction(self, id_or_reference: str) -> dict[str, Any]:
        """Fetch one transaction (``GET /transaction/:id``)."""
        id_or_reference = (id_or_reference or "").strip()
        if not id_or_reference:
            raise ConnectorError("empty transaction id or reference")
        return self._api("GET", f"/transaction/{id_or_reference}")

    def list_transactions(
        self,
        *,
        per_page: int = 50,
        page: int = 1,
        customer: str = "",
        status: str = "",
        date_from: str = "",
        date_to: str = "",
        amount: int = 0,
    ) -> dict[str, Any]:
        """Transaction history (``GET /transaction``).

        ``status``: ``success``/``failed``/``abandoned``/``pending``.
        Dates are ``YYYY-MM-DD``.
        """
        if status and status not in (
            "success", "failed", "abandoned", "pending"
        ):
            raise ConnectorError(
                f"invalid status {status!r}: use success, failed, "
                "abandoned, or pending"
            )
        params: dict[str, Any] = {
            "perPage": max(1, min(per_page, 100)),
            "page": max(1, page),
        }
        if customer:
            params["customer"] = customer
        if status:
            params["status"] = status
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        if amount:
            params["amount"] = amount
        data = self._api("GET", "/transaction", params=params)
        return {
            "transactions": data.get("data", []),
            "meta": data.get("meta", {}),
        }

    # ── customers ────────────────────────────────────────────────

    def create_customer(
        self,
        email: str,
        *,
        first_name: str = "",
        last_name: str = "",
        phone: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a customer (``POST /customer``)."""
        email = (email or "").strip()
        if not email or "@" not in email:
            raise ConnectorError(
                f"invalid customer email {email!r}"
            )
        payload: dict[str, Any] = {"email": email}
        if first_name:
            payload["first_name"] = first_name
        if last_name:
            payload["last_name"] = last_name
        if phone:
            payload["phone"] = phone
        if metadata:
            payload["metadata"] = metadata
        data = self._api("POST", "/customer", payload=payload)
        _log.info("paystack customer created: %s", email)
        return data

    def list_customers(
        self, *, per_page: int = 50, page: int = 1
    ) -> dict[str, Any]:
        """Customers (``GET /customer``)."""
        data = self._api(
            "GET", "/customer",
            params={
                "perPage": max(1, min(per_page, 100)),
                "page": max(1, page),
            },
        )
        return {"customers": data.get("data", []), "meta": data.get("meta", {})}

    # ── charging (confirmation-gated money movement) ─────────────

    def charge_authorization(
        self,
        authorization_code: str,
        email: str,
        amount_kobo: int,
        *,
        currency: str = "NGN",
        reference: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Charge a saved card (``POST /transaction/charge_authorization``).

        This pulls real money. It never runs on implied consent: pass
        ``confirmed=True`` only after the owner approved the exact
        amount/recipient — or pass ``db`` to park the exact charge on a
        human checkpoint instead.
        """
        authorization_code = (authorization_code or "").strip()
        email = (email or "").strip()
        if not authorization_code:
            raise ConnectorError("empty authorization code")
        if not email or "@" not in email:
            raise ConnectorError(f"invalid customer email {email!r}")
        if amount_kobo <= 0:
            raise ConnectorError(
                f"invalid amount {amount_kobo}: must be a positive integer "
                "in kobo"
            )
        payload = {
            "authorization_code": authorization_code,
            "email": email,
            "amount_kobo": amount_kobo,
            "currency": currency.upper(),
            "reference": reference,
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="charge",
            title=f"Charge {self._format_amount(amount_kobo, currency)}",
            instructions="\n".join([
                "Devon wants to charge a saved card via Paystack.",
                "This moves real money — review it carefully.",
                f"Amount: {self._format_amount(amount_kobo, currency)}",
                f"Customer: {email}",
                f"Reference: {reference or '(auto-generated)'}",
            ]),
            resume_state={"payload": payload},
        )
        return self._charge_now(payload)

    def _charge_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "authorization_code": payload["authorization_code"],
            "email": payload["email"],
            "amount": payload["amount_kobo"],
            "currency": payload.get("currency", "NGN"),
        }
        if payload.get("reference"):
            body["reference"] = payload["reference"]
        data = self._api("POST", "/transaction/charge_authorization",
                         payload=body)
        _log.info(
            "paystack charged %s: %s",
            payload["email"], data.get("reference", "?"),
        )
        return data

    # ── webhooks ─────────────────────────────────────────────────

    def verify_webhook_signature(
        self, raw_body: bytes, signature: str
    ) -> bool:
        """Check an incoming Paystack webhook's ``x-paystack-signature``.

        ``raw_body`` is the exact request bytes; the HMAC-SHA512 is keyed
        with the vaulted secret key. Returns False (does not raise) on
        mismatch — the receiver should answer 401 and ignore the payload.
        """
        cred = self._require_credential()
        expected = hmac.new(
            cred.password.encode("utf-8"), raw_body, hashlib.sha512
        ).hexdigest()
        return hmac.compare_digest(expected, (signature or "").strip())

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "paystack is not connected — run "
                "`nm connectors connect --name paystack` first"
            )
        return cred

    def _validate_key(self, secret: str) -> None:
        self._api("GET", "/transaction", params={"perPage": 1}, secret=secret)

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        secret: str | None = None,
    ) -> dict[str, Any]:
        """One Paystack API call; failures become PaystackError."""
        key = secret or self._require_credential().password
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {key}"}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise PaystackError(f"paystack request failed: {exc}") from exc
        if resp.status == 401:
            raise PaystackError(
                "paystack rejected the secret key (401): it is invalid or "
                "revoked — copy a fresh one from the dashboard "
                "(Settings -> API Keys & Webhooks)",
                status_code=401,
            )
        if resp.status == 429:
            raise PaystackError(
                "paystack rate limit exceeded (429) — wait and retry",
                status_code=429,
            )
        if not resp.ok:
            raise PaystackError(
                f"paystack {method} {path} failed ({resp.status}): "
                f"{self._error_text(resp)}",
                status_code=resp.status,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise PaystackError(
                f"paystack {method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(body, dict) or not body.get("status"):
            raise PaystackError(
                f"paystack {method} {path} refused: "
                f"{self._error_text(resp)}",
                status_code=resp.status,
            )
        data = body.get("data")
        if isinstance(data, dict):
            result = dict(data)
        else:
            # list endpoints return data as an array
            result = {"data": data}
        if isinstance(body.get("meta"), dict) and "meta" not in result:
            result["meta"] = body["meta"]
        return result

    @staticmethod
    def _error_text(resp: Any) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict):
                return str(body.get("message", resp.text))[:200]
        except Exception:  # noqa: BLE001 - fall back to raw text
            pass
        return resp.text[:200]

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _format_amount(amount_kobo: int, currency: str) -> str:
        major = amount_kobo / 100
        symbols = {"NGN": "₦", "GHS": "₵", "ZAR": "R", "KES": "KSh"}
        symbol = symbols.get(currency.upper(), currency.upper() + " ")
        return f"{symbol}{major:,.2f}"
