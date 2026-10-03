"""Virtual Cards connector — issue and manage virtual debit cards via API.

Provider status (verified 2026-10-03 against Flutterwave's official SDK,
``flutterwave/node-v3`` ``services/virtual-cards/*.js``, which mirrors
https://developer.flutterwave.com):

* **Implemented: Flutterwave** (https://api.flutterwave.com/v3,
  docs: https://developer.flutterwave.com). The only major
  Nigerian-serving PSP with a documented, API-driven card-issuance
  product. Auth: ``Authorization: Bearer <secret key>``.
* **Evaluated, not implemented:**
  - *Paystack*: no virtual-card issuance endpoint exists in the Paystack
    API reference (their "virtual accounts" are bank account numbers, not
    cards; no documented create/list/fund card API).
  - *Stripe Issuing*: does not serve Nigerian businesses for card
    issuance.

Response shape (per the official SDK sample responses): card objects
carry the PAN in ``card_pan``, the CVV in ``cvv``, and expiry as
``expiration`` (``"YYYY-MM"``); a pre-masked ``masked_pan`` is also
returned. Masking and secret-vaulting below key off these real field
names (``card_number``/``expiry_month``/``expiry_year`` are accepted as
legacy fallbacks only).

Structure: :class:`VirtualCardProvider` is the provider ABC — a second
provider (whenever a real, documented API exists for one) slots in by
subclassing it and registering the name in
:attr:`VirtualCardsConnector._PROVIDERS`.

Security posture (this is card data, treat it as radioactive):

* full PANs / CVVs are never logged, never returned by list/get ops, and
  never stored outside the encrypted vault;
* every outward-facing card object is masked — PAN reduced to last 4,
  CVV redacted — by :meth:`VirtualCardsConnector.mask_card`;
* the one exception is the initial create response: Flutterwave returns
  the full card details exactly once. The connector vault-stores them
  immediately (``credential_type="virtual_card"``) and returns the masked
  shape to the caller. The owner can fetch the full details later with
  :meth:`reveal_card`.

Human-in-the-loop: creating a card and funding one both move real money,
so ``provision("virtual_card", confirm=True, ...)`` and
``fund_card(..., confirm=True, ...)`` pause at a human checkpoint
(``MANUAL_STEP``) for the owner to approve the amount before any call is
made. One account per service applies to the API key, not to cards —
multiple cards per connected key are legitimate.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = [
    "FlutterwaveProvider",
    "VirtualCardProvider",
    "VirtualCardsConnector",
    "VirtualCardsError",
    "mask_card",
]

_log = get_logger(__name__)

#: Environment variable read first by connect() (non-interactive path).
ENV_VAR = "FLW_SECRET_KEY"

#: Where the owner copies their secret key from (no credentials requested
#: here — just a pointer so the connect UX can show it).
DASHBOARD_URL = "https://dashboard.flutterwave.com/settings/api"

#: Docs home for the Virtual Cards API.
DOCS_URL = "https://developer.flutterwave.com"


class VirtualCardsError(ConnectorError):
    """A virtual-card provider API call failed."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


# ── card masking ─────────────────────────────────────────────────────


def _mask_pan(pan: Any) -> Any:
    """Render a card number as last-4 only (``**** **** **** 2950``)."""
    if not isinstance(pan, str) or not pan:
        return pan
    digits = "".join(ch for ch in pan if ch.isdigit())
    if not digits:
        return "****"
    return f"**** **** **** {digits[-4:]}"


def mask_card(card: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a card object safe to log, print, or return.

    Masks the PAN (last 4 only) and redacts ``cvv``. Flutterwave's real
    field name is ``card_pan``; ``card_number`` is honored as a legacy
    fallback. Every other field passes through untouched. Works on
    Flutterwave card shapes and degrades gracefully on unknown shapes.
    """
    masked = dict(card)
    for pan_field in ("card_pan", "card_number"):
        if pan_field in masked:
            masked[pan_field] = _mask_pan(masked.get(pan_field))
    if "cvv" in masked:
        masked["cvv"] = "***" if masked.get("cvv") else masked.get("cvv")
    return masked


# ── provider layer ───────────────────────────────────────────────────


class VirtualCardProvider(ABC):
    """Provider ABC: one implementation per card-issuing PSP.

    All methods take/return plain JSON-serialisable dicts shaped like the
    provider's own API responses (envelope included). PAN masking and
    vault storage happen in the connector layer, never here.
    """

    #: registry name, e.g. "flutterwave"
    name: str = ""

    def __init__(self, http: HttpClient, api_key: str) -> None:
        if not self.name:
            raise VirtualCardsError(
                f"{type(self).__name__} must declare a provider name"
            )
        if not api_key:
            raise VirtualCardsError("virtual-card provider needs an API key")
        self.http = http
        self.api_key = api_key

    @abstractmethod
    def create_card(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Issue a card. Returns the provider's raw card object."""

    @abstractmethod
    def list_cards(self, *, per_page: int = 20) -> list[dict[str, Any]]:
        """All cards on this key (newest first where the API allows)."""

    @abstractmethod
    def get_card(self, card_id: str) -> dict[str, Any]:
        """One card by provider id."""

    @abstractmethod
    def fund_card(
        self, card_id: str, amount: float, *, debit_currency: str
    ) -> dict[str, Any]:
        """Move money from the merchant balance onto the card."""

    @abstractmethod
    def withdraw_card(self, card_id: str, amount: float) -> dict[str, Any]:
        """Pull money off the card back to the merchant balance."""

    @abstractmethod
    def terminate_card(self, card_id: str) -> dict[str, Any]:
        """Permanently terminate the card."""

    @abstractmethod
    def block_card(self, card_id: str, action: str) -> dict[str, Any]:
        """``action`` is ``"block"`` or ``"unblock"``."""

    @abstractmethod
    def list_transactions(
        self,
        card_id: str,
        *,
        from_date: str = "",
        to_date: str = "",
        index: int = 0,
        size: int = 20,
    ) -> dict[str, Any]:
        """Card transaction ledger (provider's envelope)."""


class FlutterwaveProvider(VirtualCardProvider):
    """Flutterwave Virtual Cards (api.flutterwave.com/v3).

    Endpoints (Bearer secret key) — verified 2026-10-03 against the
    official ``flutterwave/node-v3`` SDK source
    (``services/virtual-cards/*.js``):

    * ``POST   /v3/virtual-cards`` — create
    * ``GET    /v3/virtual-cards`` — list
    * ``GET    /v3/virtual-cards/:id`` — fetch one
    * ``POST   /v3/virtual-cards/:id/fund`` — fund
      (payload ``{"debit_currency": ..., "amount": ...}``)
    * ``POST   /v3/virtual-cards/:id/withdraw`` — withdraw
      (payload ``{"amount": ...}``)
    * ``PUT    /v3/virtual-cards/:id/terminate`` — terminate
    * ``GET    /v3/virtual-cards/:id/transactions?from=&to=&index=&size=``
    * ``PUT    /v3/virtual-cards/:id/status/:block|:unblock`` — block/unblock
      (the action rides in the path, per the SDK; the payload passes
      through too)

    The API answers with ``{"status": "success"|"error", "message": str,
    "data": ...}``.
    """

    name = "flutterwave"
    API_BASE = "https://api.flutterwave.com/v3"

    # ── create payload shape ───────────────────────────────────────

    #: fields accepted by POST /v3/virtual-cards (per the official SDK's
    #: create-card payload example — note ``debit_currency``, the currency
    #: the funding is debited in, is required alongside currency/amount).
    CREATE_FIELDS = (
        "currency",
        "amount",
        "debit_currency",
        "billing_name",
        "billing_address",
        "billing_city",
        "billing_state",
        "billing_postal_code",
        "billing_country",
        "first_name",
        "last_name",
        "date_of_birth",
        "email",
        "phone",
        "title",
        "gender",
        "callback_url",
    )

    def create_card(self, payload: dict[str, Any]) -> dict[str, Any]:
        filtered = {
            k: v for k, v in payload.items() if k in self.CREATE_FIELDS
        }
        if not filtered.get("currency") or not filtered.get("amount"):
            raise VirtualCardsError(
                "create_card needs at least 'currency' and 'amount'"
            )
        data = self._api("POST", "/virtual-cards", filtered)
        return self._card(data)

    def list_cards(self, *, per_page: int = 20) -> list[dict[str, Any]]:
        data = self._api(
            "GET", "/virtual-cards", params={"per_page": max(1, per_page)}
        )
        cards = data if isinstance(data, list) else []
        return [self._card(c) for c in cards]

    def get_card(self, card_id: str) -> dict[str, Any]:
        return self._card(self._api("GET", f"/virtual-cards/{card_id}"))

    def fund_card(
        self, card_id: str, amount: float, *, debit_currency: str
    ) -> dict[str, Any]:
        _require_amount(amount, "fund")
        data = self._api(
            "POST",
            f"/virtual-cards/{card_id}/fund",
            {"debit_currency": debit_currency, "amount": amount},
        )
        return data if isinstance(data, dict) else {"data": data}

    def withdraw_card(self, card_id: str, amount: float) -> dict[str, Any]:
        _require_amount(amount, "withdraw")
        data = self._api(
            "POST",
            f"/virtual-cards/{card_id}/withdraw",
            {"amount": amount},
        )
        return data if isinstance(data, dict) else {"data": data}

    def terminate_card(self, card_id: str) -> dict[str, Any]:
        data = self._api("PUT", f"/virtual-cards/{card_id}/terminate", {})
        return data if isinstance(data, dict) else {"data": data}

    def block_card(self, card_id: str, action: str) -> dict[str, Any]:
        if action not in ("block", "unblock"):
            raise VirtualCardsError(
                f"block_card action must be 'block' or 'unblock', "
                f"got {action!r}"
            )
        # Per the official SDK (rave.block_unblock.js): the action is part
        # of the path — PUT v3/virtual-cards/{id}/status/{block|unblock}.
        data = self._api(
            "PUT",
            f"/virtual-cards/{card_id}/status/{action}",
            {"status_action": action},
        )
        return data if isinstance(data, dict) else {"data": data}

    def list_transactions(
        self,
        card_id: str,
        *,
        from_date: str = "",
        to_date: str = "",
        index: int = 0,
        size: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"index": index, "size": max(1, size)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self._api(
            "GET", f"/virtual-cards/{card_id}/transactions", params=params
        )
        if isinstance(data, dict):
            return data
        return {"transactions": data}

    # ── HTTP plumbing ────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=self._headers(), params=params
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=self._headers()
                )
            elif method == "PUT":
                resp = self.http.put_json(
                    url, payload or {}, headers=self._headers()
                )
            else:
                raise VirtualCardsError(f"unsupported method {method}")
        except VirtualCardsError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise VirtualCardsError(
                f"flutterwave request failed: {exc}"
            ) from exc
        return self._unwrap(method, path, resp)

    def _unwrap(self, method: str, path: str, resp: Any) -> Any:
        if resp.status == 401:
            raise VirtualCardsError(
                "flutterwave rejected the secret key (401): it is invalid, "
                "expired, or revoked — reconnect with a fresh key from "
                f"{DASHBOARD_URL}",
                status_code=401,
            )
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            body = {}
        if isinstance(body, dict) and body.get("status") == "error":
            message = str(body.get("message", ""))
            raise VirtualCardsError(
                f"flutterwave {method} {path} failed "
                f"({resp.status}): {message}",
                status_code=resp.status,
            )
        if not resp.ok:
            text = getattr(resp, "text", "")[:200]
            raise VirtualCardsError(
                f"flutterwave {method} {path} failed "
                f"({resp.status}): {text}",
                status_code=resp.status,
            )
        if isinstance(body, dict):
            return body.get("data", body)
        return body

    @staticmethod
    def _card(data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            return data
        raise VirtualCardsError(
            f"flutterwave returned an unexpected card shape: {data!r}"
        )


def _require_amount(amount: float, verb: str) -> None:
    try:
        value = float(amount)
    except (TypeError, ValueError) as exc:
        raise VirtualCardsError(
            f"cannot {verb}: amount {amount!r} is not a number"
        ) from exc
    if value <= 0:
        raise VirtualCardsError(f"cannot {verb}: amount must be > 0")


# ── connector ──────────────────────────────────────────────────────


@register_connector
class VirtualCardsConnector(Connector):
    """Devon's virtual-card adapter (Flutterwave primary)."""

    id = "virtualcards"
    name = "Virtual Cards"
    description = (
        "Issue and manage virtual debit cards via Flutterwave "
        "(create, list, fund, withdraw, block/unblock, terminate, "
        "transactions). Authenticates with a Flutterwave secret key. "
        "Card numbers are masked everywhere except a one-time vault copy "
        "at creation."
    )
    auth_methods = (AuthMethod.API_KEY,)
    PROVISIONABLE = ("virtual_card",)

    #: provider name -> provider class. A second provider slots in here.
    _PROVIDERS: dict[str, type[VirtualCardProvider]] = {
        FlutterwaveProvider.name: FlutterwaveProvider,
    }

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        api_key: str | None = None,
        provider: str = "flutterwave",
    ) -> ConnectResult:
        """Store the secret key after validating it against the API.

        Key arrives by argument, the ``FLW_SECRET_KEY`` env var, or a secure
        TTY prompt (env-first). Use a ``FLWSECK_TEST-...`` key for sandbox.
        """
        key = (api_key or "").strip()
        if not key and not os.environ.get(ENV_VAR, "").strip():
            print(self.connect_instructions())
        key = key or prompt_secret("Flutterwave secret key", env_var=ENV_VAR)
        if not key:
            raise ConnectorError("empty key: nothing to connect with")
        provider_cls = self._provider_cls(provider)
        flw = provider_cls(self.http, key)
        # Lightweight validation call: a bad key answers 401.
        cards = flw.list_cards(per_page=1)
        cred = self._store_credential(
            provider,
            key,
            credential_type="api_key",
            scopes=["virtual_cards"],
            metadata={"provider": provider, "cards_seen": len(cards)},
        )
        _log.info("virtualcards connected (provider %s)", provider)
        return ConnectResult(
            ok=True,
            account=provider,
            scopes=["virtual_cards"],
            message=(
                f"connected to {provider} virtual cards. The secret key is "
                "in the encrypted vault; revoke/rotate it any time at "
                f"{DASHBOARD_URL}."
            ),
            credential_id=cred.id,
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail=(
                    "not connected — run "
                    "`nm connectors connect --name virtualcards`"
                ),
            )
        try:
            cards = self._provider(cred).list_cards(per_page=1)
        except VirtualCardsError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"secret key rejected ({exc}): reconnect with a "
                "fresh key",
            )
        return ConnectorStatus(
            connected=True,
            account=cred.username,
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"secret key valid ({len(cards)} card(s) visible on "
            "first page)",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._provider(cred).list_cards(per_page=1)
            return True
        except ConnectorError:
            return False

    # ── provisioning ───────────────────────────────────────────

    def provision(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        """Provision ``virtual_card`` on the owner's request/standing permission.

        ``confirm=True`` (with ``db=``) pauses at a human checkpoint so the
        owner approves the funding amount before any money moves.
        """
        if kind != "virtual_card":
            raise ConnectorError(
                f"virtualcards cannot provision {kind!r} "
                f"(provisionable: {', '.join(self.PROVISIONABLE)})"
            )
        db = kwargs.pop("db", None)
        context = kwargs.pop("context", None)
        confirm = kwargs.pop("confirm", False)
        if confirm:
            if db is None:
                raise ConnectorError(
                    "virtual_card confirmation needs a database for "
                    "checkpoints (pass db=)"
                )
            self._validate_create_payload(kwargs)
            amount = kwargs.get("amount")
            currency = kwargs.get("currency")
            from .checkpoints import CheckpointKind

            cp = self.request_human(
                CheckpointKind.MANUAL_STEP,
                f"Create virtual card — approve {currency} {amount}",
                "\n".join(
                    [
                        "Devon is about to create a virtual card, which "
                        "moves real money:",
                        f"  amount:   {currency} {amount}",
                        f"  name:     {kwargs.get('billing_name', '')}",
                        "This is YOUR money on YOUR Flutterwave account — "
                        "nothing happens until you approve.",
                    ]
                ),
                db=db,
                context=context,
                resume_state={
                    "intent": "create_card",
                    "create_kwargs": kwargs,
                },
            )
            # Interactive: resolved already — continue through.
            return self.resume_checkpoint(cp, db=db, context=context)
        return self.create_card(**kwargs)

    # ── card operations (masked output) ──────────────────────────

    def create_card(self, **kwargs: Any) -> dict[str, Any]:
        """Issue a card; returns the *masked* card object.

        Required: ``currency`` (e.g. ``"USD"`` — the usual live denomination
        for Nigerian businesses — or ``"NGN"`` in test), ``amount``, plus
        the billing fields Flutterwave expects (``billing_name``,
        ``billing_address``, ``billing_city``, ``billing_state``,
        ``billing_postal_code``, ``billing_country``; for USD cards also
        ``first_name``, ``last_name``, ``date_of_birth`` (``YYYY/MM/DD``),
        ``email``, ``phone``, ``title``, ``gender``).

        The full card details come back exactly once in the API response;
        the connector vault-stores them immediately and returns only the
        masked shape (last 4). Fetch them later with :meth:`reveal_card`.
        """
        self._validate_create_payload(kwargs)
        card = self._require_provider().create_card(kwargs)
        card_id = str(card.get("id", ""))
        self._store_card_secrets(card_id, card)
        masked = mask_card(card)
        _log.info(
            "virtualcards card created: id=%s last4=%s",
            card_id,
            _last4(card),
        )
        return masked

    def list_cards(self, *, per_page: int = 20) -> list[dict[str, Any]]:
        """Every card on the key — masked (PAN last 4 only, no CVV)."""
        cards = self._require_provider().list_cards(per_page=per_page)
        return [mask_card(c) for c in cards]

    def get_card(self, card_id: str) -> dict[str, Any]:
        """One card — masked (PAN last 4 only, no CVV)."""
        return mask_card(self._require_provider().get_card(card_id))

    def fund_card(
        self,
        card_id: str,
        amount: float,
        *,
        debit_currency: str = "NGN",
        confirm: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Fund a card from the merchant balance.

        ``confirm=True`` (with ``db=``) pauses at a human checkpoint for
        owner approval of the amount before money moves.
        """
        _require_amount(amount, "fund")
        if confirm:
            if db is None:
                raise ConnectorError(
                    "fund_card confirmation needs a database for "
                    "checkpoints (pass db=)"
                )
            from .checkpoints import CheckpointKind

            cp = self.request_human(
                CheckpointKind.MANUAL_STEP,
                f"Fund card {card_id} — approve {debit_currency} {amount}",
                "\n".join(
                    [
                        "Devon is about to move real money onto a card:",
                        f"  card:     {card_id}",
                        f"  amount:   {debit_currency} {amount}",
                        "This debits YOUR Flutterwave balance — nothing "
                        "happens until you approve.",
                    ]
                ),
                db=db,
                context=context,
                resume_state={
                    "intent": "fund_card",
                    "card_id": card_id,
                    "amount": amount,
                    "debit_currency": debit_currency,
                },
            )
            return self.resume_checkpoint(cp, db=db, context=context)
        result = self._require_provider().fund_card(
            card_id, float(amount), debit_currency=debit_currency
        )
        _log.info(
            "virtualcards card %s funded: %s %s", card_id, debit_currency,
            amount,
        )
        return result

    def withdraw_card(
        self, card_id: str, amount: float
    ) -> dict[str, Any]:
        """Pull funds off the card back to the merchant balance."""
        _require_amount(amount, "withdraw")
        result = self._require_provider().withdraw_card(
            card_id, float(amount)
        )
        _log.info("virtualcards card %s withdrew %s", card_id, amount)
        return result

    def block_card(self, card_id: str) -> dict[str, Any]:
        """Temporarily block the card (reversible with unblock_card)."""
        return self._require_provider().block_card(card_id, "block")

    def unblock_card(self, card_id: str) -> dict[str, Any]:
        """Unblock a previously blocked card."""
        return self._require_provider().block_card(card_id, "unblock")

    def terminate_card(self, card_id: str) -> dict[str, Any]:
        """Permanently terminate the card. Not reversible."""
        result = self._require_provider().terminate_card(card_id)
        self._drop_card_secrets(card_id)
        _log.info("virtualcards card %s terminated", card_id)
        return result

    def list_transactions(
        self,
        card_id: str,
        *,
        from_date: str = "",
        to_date: str = "",
        index: int = 0,
        size: int = 20,
    ) -> dict[str, Any]:
        """Ledger entries for one card (masked PANs in nested card refs)."""
        result = self._require_provider().list_transactions(
            card_id,
            from_date=from_date,
            to_date=to_date,
            index=index,
            size=size,
        )
        return self._mask_nested(result)

    def reveal_card(self, card_id: str) -> dict[str, Any]:
        """Return the full card details from the vault to the owner.

        This is the only path that surfaces a complete PAN/CVV — it reads
        the vault copy saved at creation time. The caller hands it to the
        owner directly; it is never logged and never returned by list/get.
        Fails fast when the card was not created through this connector.
        """
        from ..core.errors import NotFound

        try:
            cred = self.vault.get(self._service, f"card:{card_id}")
        except NotFound:
            raise ConnectorError(
                f"no stored card secrets for {card_id!r} — the card was "
                "not created through this connector (or its secrets were "
                "dropped after termination)"
            ) from None
        try:
            details = json.loads(cred.password)
        except (ValueError, TypeError) as exc:
            raise ConnectorError(
                f"stored card secrets for {card_id!r} are corrupt"
            ) from exc
        _log.info("virtualcards card %s revealed to owner", card_id)
        return details

    # ── checkpoint resume ──────────────────────────────────────

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue a create/fund flow after the owner approved the amount."""
        from .checkpoints import CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve the amount first"
            )
        state = checkpoint.resume_state or {}
        intent = state.get("intent", "")
        if intent == "create_card":
            return self.create_card(**(state.get("create_kwargs") or {}))
        if intent == "fund_card":
            return self.fund_card(
                state.get("card_id", ""),
                state.get("amount", 0),
                debit_currency=state.get("debit_currency", "NGN"),
            )
        raise ConnectorError(
            f"virtualcards cannot resume checkpoint intent {intent!r}"
        )

    # ── internals ──────────────────────────────────────────────

    @classmethod
    def _provider_cls(cls, provider: str) -> type[VirtualCardProvider]:
        try:
            return cls._PROVIDERS[provider]
        except KeyError:
            known = ", ".join(sorted(cls._PROVIDERS))
            raise ConnectorError(
                f"unknown virtual-card provider {provider!r} "
                f"(known: {known})"
            ) from None

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "virtualcards is not connected — run "
                "`nm connectors connect --name virtualcards` first"
            )
        return cred

    def _provider(self, cred: Any) -> VirtualCardProvider:
        provider = (cred.metadata or {}).get("provider", "flutterwave")
        return self._provider_cls(provider)(self.http, cred.password)

    def _require_provider(self) -> VirtualCardProvider:
        return self._provider(self._require_credential())

    @staticmethod
    def _validate_create_payload(kwargs: dict[str, Any]) -> None:
        missing = [
            f for f in ("currency", "amount") if not kwargs.get(f)
        ]
        if missing:
            raise ConnectorError(
                "create_card needs 'currency' and 'amount' "
                f"(missing: {', '.join(missing)})"
            )
        _require_amount(kwargs["amount"], "create")
        if not kwargs.get("billing_name"):
            raise ConnectorError(
                "create_card needs 'billing_name' (cardholder name)"
            )

    # ── card-secret vault plumbing ─────────────────────────────

    #: Card-secret fields copied into the vault at creation. Flutterwave's
    #: real field names first; ``card_number``/``expiry_month``/
    #: ``expiry_year`` kept as legacy fallbacks only.
    _CARD_SECRET_FIELDS = (
        "card_pan",
        "cvv",
        "expiration",
        "masked_pan",
        "card_id",
        "id",
        "card_number",
        "expiry_month",
        "expiry_year",
    )

    def _store_card_secrets(
        self, card_id: str, card: dict[str, Any]
    ) -> None:
        """Vault the full card details once, at creation. Never logged."""
        details = {
            k: card[k]
            for k in self._CARD_SECRET_FIELDS
            if card.get(k) is not None
        }
        if "card_id" not in details and card_id:
            details["card_id"] = card_id
        self.vault.store(
            service=self._service,
            username=f"card:{card_id}",
            password=json.dumps(details),
            credential_type="virtual_card",
            tags=["connector", self.id, "virtual_card"],
            metadata={"card_id": card_id, "masked": True},
        )
        _log.info("virtualcards card secrets vault-stored: id=%s", card_id)

    def _drop_card_secrets(self, card_id: str) -> None:
        """Remove stored card secrets (termination kills the card)."""
        from ..core.errors import NotFound

        with contextlib.suppress(NotFound):
            self.vault.delete(self._service, f"card:{card_id}")
            _log.info(
                "virtualcards card secrets dropped: id=%s", card_id
            )

    @staticmethod
    def _mask_nested(data: Any) -> Any:
        """Mask card objects nested anywhere in a transactions envelope."""
        if isinstance(data, dict):
            if ("card_pan" in data or "card_number" in data
                    or "cvv" in data):
                return mask_card(data)
            return {k: VirtualCardsConnector._mask_nested(v)
                    for k, v in data.items()}
        if isinstance(data, list):
            return [VirtualCardsConnector._mask_nested(v) for v in data]
        return data

    def connect_instructions(self) -> str:
        """The human steps for the part the API cannot do: get the key."""
        return "\n".join(
            [
                "1. Open " + DASHBOARD_URL + " and log in (your own account).",
                "2. Copy your secret key "
                "(test keys start with FLWSECK_TEST- for sandbox).",
                "3. Paste it below (or set the " + ENV_VAR
                + " environment variable).",
                "Full API docs: " + DOCS_URL,
            ]
        )


def _last4(card: dict[str, Any]) -> str:
    pan = card.get("card_pan") or card.get("card_number") or ""
    digits = "".join(ch for ch in str(pan) if ch.isdigit())
    return digits[-4:] if digits else "?"
