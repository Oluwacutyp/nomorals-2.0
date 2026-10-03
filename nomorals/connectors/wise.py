"""Wise connector — international money transfers.

Docs: https://docs.wise.com/

Auth: API token (``AuthMethod.API_KEY``) as a Bearer token, from the
Wise dashboard (Settings → API tokens). Tokens are scoped by Wise —
the transfer endpoints need the ``transfers`` scope.

The money flow is quote → recipient → transfer → funding:
``create_quote`` (``POST /v2/quotes``) is the price-comparison quote;
``create_authenticated_quote`` (``POST /v3/profiles/{id}/quotes``) locks
a rate against your profile and is what real transfers are built on;
``create_transfer`` (``POST /v1/transfers``) stages the transfer
(``targetAccount`` is a recipient account id from ``GET /v2/accounts``);
the transfer sits in ``waiting_for_funds`` until funded — funding
(``POST /v3/profiles/{id}/transfers/{id}/payments``) is NOT implemented
yet, so fund from the Wise app/dashboard after creating a transfer.

REAL MONEY: ``create_transfer`` never runs on implied consent — it
requires ``confirmed=True`` (owner approved the exact quote/recipient)
or a human checkpoint when ``db`` is given. A generated UUID
``customerTransactionId`` makes creation idempotent: retrying with the
same triple returns the existing transfer instead of a duplicate.

Sandbox: pass ``sandbox=True`` to connect() to point at
``https://api.sandbox.transferwise.tech`` with a sandbox token.
"""

from __future__ import annotations

import time
import uuid
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

__all__ = ["WiseConnector", "WiseError"]

_log = get_logger(__name__)

API_BASE = "https://api.wise.com"
SANDBOX_BASE = "https://api.sandbox.transferwise.tech"
TOKEN_ENV = "WISE_API_TOKEN"


class WiseError(ConnectorError):
    """A Wise API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, error_code: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@register_connector
class WiseConnector(Connector):
    """Devon's Wise adapter: profiles, quotes, and transfers."""

    id = "wise"
    name = "Wise"
    description = (
        "Wise money transfers: list profiles, create FX quotes, and "
        "create transfers (Bearer API token; transfers require explicit "
        "owner confirmation). Sandbox supported."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        sandbox: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate an API token against /v1/profiles and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "wise is already connected — one account per service. "
                "Disconnect first to switch API tokens."
            )
        secret = (token or "").strip() or prompt_secret(
            "Wise API token", env_var=TOKEN_ENV
        )
        if not secret:
            raise ConnectorError("empty API token: nothing to connect with")
        base = SANDBOX_BASE if sandbox else API_BASE
        profiles = self._api(
            "GET", "/v1/profiles", token=secret, base_url=base
        )
        if not isinstance(profiles, list):
            raise WiseError("wise /v1/profiles returned an unexpected shape")
        names = ", ".join(
            p.get("fullName") or p.get("name") or str(p.get("id"))
            for p in profiles
        )
        self._store_credential(
            "wise",
            secret,
            credential_type="api_key",
            scopes=["profiles:read", "quotes", "transfers"],
            metadata={"base_url": base, "sandbox": sandbox},
        )
        _log.info(
            "wise connected (%s, %d profile(s))",
            "sandbox" if sandbox else "live", len(profiles),
        )
        return ConnectResult(
            ok=True,
            account=f"wise ({'sandbox' if sandbox else 'live'})",
            scopes=["profiles:read", "quotes", "transfers"],
            message=(
                f"connected to Wise {'sandbox' if sandbox else 'LIVE'} "
                f"({len(profiles)} profile(s): {names or 'none'}). The "
                "token is in the encrypted vault. Every transfer still "
                "needs your explicit confirmation at call time."
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
                       "--name wise`",
            )
        try:
            profiles = self._api("GET", "/v1/profiles")
        except WiseError as exc:
            return ConnectorStatus(
                connected=False,
                account="wise",
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): rotate it in the Wise "
                       "dashboard and reconnect",
            )
        mode = "sandbox" if (cred.metadata or {}).get("sandbox") else "live"
        return ConnectorStatus(
            connected=True,
            account=f"wise ({mode})",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"token valid; "
                   f"{len(profiles) if isinstance(profiles, list) else '?'} "
                   f"profile(s)",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/v1/profiles")
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
        """Execute a confirmed transfer after the owner resolved it."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve the transfer first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        payload = dict((checkpoint.resume_state or {}).get("payload", {}))
        if stage == "create_transfer":
            if not payload.get("quote_uuid"):
                raise WiseError(
                    "the resolved checkpoint has no transfer payload — "
                    "it cannot create the transfer"
                )
            return self._create_now(payload)
        raise WiseError(
            f"wise cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def list_profiles(self) -> list[dict[str, Any]]:
        """Profiles on the token (``GET /v1/profiles``)."""
        data = self._api("GET", "/v1/profiles")
        if not isinstance(data, list):
            raise WiseError("wise /v1/profiles returned an unexpected shape")
        return data

    def get_transfer(self, transfer_id: int) -> dict[str, Any]:
        """One transfer (``GET /v1/transfers/{id}``) — status, amounts."""
        if not transfer_id:
            raise ConnectorError("transfer_id is required")
        data = self._api("GET", f"/v1/transfers/{transfer_id}")
        return data if isinstance(data, dict) else {}

    def list_recipients(self, profile_id: int) -> list[dict[str, Any]]:
        """Recipient accounts on a profile (``GET /v2/accounts``).

        The ``id`` of an entry is the ``target_account`` that
        ``create_transfer`` takes.
        """
        if not profile_id:
            raise ConnectorError("profile_id is required")
        data = self._api(
            "GET", "/v2/accounts",
            params={"profile": profile_id, "type": "list"},
        )
        return data if isinstance(data, list) else []

    # ── quotes ───────────────────────────────────────────────────

    def create_quote(
        self,
        source_currency: str,
        target_currency: str,
        *,
        source_amount: float = 0.0,
        target_amount: float = 0.0,
    ) -> dict[str, Any]:
        """A price-comparison quote (``POST /v2/quotes``).

        Give exactly one of ``source_amount`` / ``target_amount``.
        Unauthenticated quotes are for comparing rates — they cannot back
        a real transfer; use ``create_authenticated_quote`` for money.
        """
        quote = self._quote_body(
            source_currency, target_currency, source_amount, target_amount
        )
        data = self._api("POST", "/v2/quotes", body=quote)
        return data if isinstance(data, dict) else {}

    def create_authenticated_quote(
        self,
        profile_id: int,
        source_currency: str,
        target_currency: str,
        *,
        source_amount: float = 0.0,
        target_amount: float = 0.0,
    ) -> dict[str, Any]:
        """A rate-locked quote for a real transfer
        (``POST /v3/profiles/{id}/quotes``).

        The returned quote's ``id`` (UUID) is the ``quote_uuid`` that
        ``create_transfer`` takes. The rate is locked for a limited time
        (see ``rateExpirationTime`` in the response).
        """
        if not profile_id:
            raise ConnectorError("profile_id is required")
        quote = self._quote_body(
            source_currency, target_currency, source_amount, target_amount
        )
        data = self._api(
            "POST", f"/v3/profiles/{profile_id}/quotes", body=quote
        )
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _quote_body(
        source_currency: str,
        target_currency: str,
        source_amount: float,
        target_amount: float,
    ) -> dict[str, Any]:
        source_currency = (source_currency or "").upper()
        target_currency = (target_currency or "").upper()
        if not source_currency or not target_currency:
            raise ConnectorError(
                "source_currency and target_currency are required "
                "(ISO 4217, e.g. USD, EUR)"
            )
        if (source_amount > 0) == (target_amount > 0):
            raise ConnectorError(
                "pass exactly one of source_amount / target_amount"
            )
        body: dict[str, Any] = {
            "sourceCurrency": source_currency,
            "targetCurrency": target_currency,
        }
        if source_amount > 0:
            body["sourceAmount"] = source_amount
        else:
            body["targetAmount"] = target_amount
        return body

    # ── transfers (confirmation-gated real money) ────────────────

    def create_transfer(
        self,
        target_account: int,
        quote_uuid: str,
        *,
        customer_transaction_id: str = "",
        reference: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Stage a transfer (``POST /v1/transfers``). REAL MONEY.

        Never runs on implied consent: pass ``confirmed=True`` only after
        the owner approved the exact quote and recipient — or pass ``db``
        to park the exact payload on a human checkpoint.
        ``customer_transaction_id`` is the idempotency key; a random UUID
        is generated when omitted, so retries never double-create.
        ``target_account`` is a recipient id from ``list_recipients``.
        The transfer lands in ``waiting_for_funds`` — fund it from the
        Wise app/dashboard (funding via API is not implemented yet).
        """
        if not target_account:
            raise ConnectorError(
                "target_account is required — a recipient id from "
                "list_recipients()"
            )
        quote_uuid = (quote_uuid or "").strip()
        if not quote_uuid:
            raise ConnectorError(
                "quote_uuid is required — from create_authenticated_quote()"
            )
        payload: dict[str, Any] = {
            "target_account": target_account,
            "quote_uuid": quote_uuid,
            "customer_transaction_id": (
                customer_transaction_id.strip() or str(uuid.uuid4())
            ),
            "reference": reference.strip(),
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="create_transfer",
            title="Create Wise transfer",
            instructions="\n".join([
                "Devon wants to stage this REAL-MONEY transfer.",
                "Review it — once funded it moves money.",
                f"Quote: {quote_uuid}",
                f"Recipient account id: {target_account}",
                f"Reference: {reference}" if reference else "",
                f"Idempotency key: {payload['customer_transaction_id']}",
            ]),
            resume_state={"payload": payload},
        )
        return self._create_now(payload)

    def _create_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "targetAccount": payload["target_account"],
            "quoteUuid": payload["quote_uuid"],
            "customerTransactionId": payload["customer_transaction_id"],
        }
        if payload.get("reference"):
            body["details"] = {"reference": payload["reference"]}
        data = self._api("POST", "/v1/transfers", body=body)
        result = data if isinstance(data, dict) else {}
        _log.info(
            "wise transfer created: id %s (status %s)",
            result.get("id", "?"), result.get("status", "?"),
        )
        return result

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "wise is not connected — run "
                "`nm connectors connect --name wise` first"
            )
        return cred

    def _base_url(self) -> str:
        cred = self._load_credential()
        meta = (cred.metadata or {}) if cred else {}
        return str(meta.get("base_url", API_BASE))

    def _api(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        token: str = "",
        base_url: str = "",
    ) -> Any:
        """One Wise API call. Failures become WiseError."""
        secret = token or self._require_credential().password
        base = base_url or self._base_url()
        url = f"{base}{path}"
        headers = {"Authorization": f"Bearer {secret}"}
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=headers, params=params or None
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, body or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise WiseError(f"wise request failed: {exc}") from exc
        if resp.status == 401:
            raise WiseError(
                "wise rejected the API token (401): it is invalid or "
                "revoked — create a fresh one in the Wise dashboard and "
                "reconnect",
                status_code=401,
            )
        if resp.status == 403:
            raise WiseError(
                "wise refused (403): the token lacks the scope for this "
                "endpoint — check the token's permissions in the Wise "
                "dashboard",
                status_code=403,
            )
        if resp.status == 429:
            raise WiseError(
                "wise rate limit hit (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            code, msg = self._error_detail(resp)
            raise WiseError(
                f"wise {method} {path} failed ({resp.status}): {msg}",
                status_code=resp.status,
                error_code=code,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise WiseError(
                f"wise {method} {path} returned invalid JSON"
            ) from exc

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, str]:
        try:
            body = resp.json()
            if isinstance(body, dict):
                errors = body.get("errors", [])
                if errors:
                    first = errors[0]
                    if isinstance(first, dict):
                        return (
                            str(first.get("code", "")),
                            str(first.get("message", body))[:200],
                        )
                return "", str(
                    body.get("message", body.get("error", body))
                )[:200]
        except Exception:  # noqa: BLE001 - fall back to raw text
            pass
        return "", resp.text[:200]
