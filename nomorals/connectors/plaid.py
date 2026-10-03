"""Plaid connector — US/EU bank data + money movement.

Plaid (https://plaid.com, docs: https://plaid.com/docs/api) aggregates bank
accounts, transactions, liabilities, and investments behind one API, and —
with the Transfer and Payment Initiation products — moves money too.
Auth is a ``client_id`` + ``secret`` pair sent in every JSON request body,
plus one ``access_token`` per linked bank item.

Linking a bank is a human-in-the-loop flow: Devon mints a Plaid Link token
(``POST /link/token/create``), the owner completes Plaid Link in their own
browser (entering their own bank credentials there — Devon never sees
them), Link hands back a ``public_token``, and Devon exchanges it for a
long-lived ``access_token`` (``POST /item/public_token/exchange``).

FULL FUNCTIONALITY. This connector reads bank data AND moves money:
US ACH transfers via the Transfer API (``/transfer/...``) and UK/EU
payments via Payment Initiation (``/payment_initiation/...``). Every
money-moving call takes ``confirm=True`` to pause at a human checkpoint
so the owner approves the exact amount, currency, and destination before
anything moves — the owner confirms, Devon never blocks. All movements
are idempotency-keyed and written to a JSONL audit log.

Rules honored from the Plaid playbook:
* never expose full account or routing numbers — Plaid only returns masked
  numbers (last digits) and the summaries keep it that way;
* plain-English summaries: no command names, field names, cursors, or
  internal ids in owner-facing output;
* Plaid data can lag the bank — summaries say so instead of asserting
  completeness.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

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

__all__ = ["PlaidConnector", "PlaidError"]

_log = get_logger(__name__)

BASE_URLS = {
    "sandbox": "https://sandbox.plaid.com",
    "production": "https://production.plaid.com",
}
CLIENT_ID_ENV = "PLAID_CLIENT_ID"
SECRET_ENV = "PLAID_SECRET"
ACCESS_TOKEN_ENV = "PLAID_ACCESS_TOKEN"
ENV_ENV = "PLAID_ENV"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Plaid error codes mapped to owner-actionable guidance.
_ERROR_GUIDANCE = {
    "INVALID_ACCESS_TOKEN": (
        "the access token is invalid or revoked — re-link the bank"
    ),
    "ITEM_LOGIN_REQUIRED": (
        "the bank needs the owner to sign in again — re-link through "
        "Plaid Link"
    ),
    "ITEM_LOCKED": "the bank locked the item — the owner must unlock it "
                   "at the bank, then re-link",
    "RATE_LIMIT_EXCEEDED": "plaid rate limit hit — wait a minute and retry",
    "INSTITUTION_NOT_RESPONDING": "the bank is not responding to Plaid "
                                  "right now — retry later",
    "INSTITUTION_DOWN": "the bank's Plaid integration is down — retry later",
    "INVALID_API_KEYS": "client_id/secret rejected — check the pair in the "
                        "Plaid dashboard",
    "TRANSFER_NOT_ENABLED": "Plaid Transfer is not enabled on this account "
                            "— enable the Transfer product in the Plaid "
                            "dashboard first",
    "PRODUCT_NOT_ENABLED": "that Plaid product is not enabled on this "
                           "account — enable it in the Plaid dashboard",
}


class PlaidError(ConnectorError):
    """A Plaid API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        error_code: str = "",
        error_type: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type


@register_connector
class PlaidConnector(Connector):
    """Devon's Plaid adapter: bank data, US ACH transfers, UK/EU payments.

    Reads accounts, transactions, liabilities, and investments; moves
    money via the Transfer API (US ACH) and Payment Initiation (UK/EU).
    Money-moving calls take ``confirm=True`` to pause at a human
    checkpoint for owner approval.
    """

    id = "plaid"
    name = "Plaid"
    description = (
        "US/EU bank data through Plaid plus money movement: accounts and "
        "balances, transaction history and sync, recurring streams, "
        "liabilities, investment holdings/transactions, US ACH transfers "
        "via the Transfer API, and UK/EU payments via Payment Initiation. "
        "Authenticates with a client_id + secret pair and one access token "
        "per linked bank. Transfers and payments take confirm=True for "
        "owner approval before money moves."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        client_id: str | None = None,
        secret: str | None = None,
        env: str | None = None,
        access_token: str | None = None,
        institution_name: str = "",
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Store the Plaid client_id + secret; optionally link a bank now.

        Credentials arrive via arguments, the ``PLAID_CLIENT_ID`` /
        ``PLAID_SECRET`` env vars, or a secure prompt, and are validated
        against the API before anything is vault-stored. ``env`` selects
        ``sandbox`` (default) or ``production``. When ``access_token`` (or
        the ``PLAID_ACCESS_TOKEN`` env var) is given, that bank item is
        linked in the same step. Otherwise the owner links through Plaid
        Link — see :meth:`begin_link`.
        """
        if self._client_credential() is not None:
            raise ConnectorError(
                "plaid is already connected — one account per service. "
                "Disconnect first to switch client credentials."
            )
        cid = (client_id or "").strip() or prompt_secret(
            "Plaid client_id", env_var=CLIENT_ID_ENV
        )
        sec = (secret or "").strip() or prompt_secret(
            "Plaid secret", env_var=SECRET_ENV
        )
        if not cid or not sec:
            raise ConnectorError(
                "empty client_id/secret: nothing to connect with"
            )
        environment = self._normalize_env(env)
        base = BASE_URLS[environment]
        self._validate_client(base, cid, sec)
        self._store_credential(
            "__client__",
            json.dumps({"client_id": cid, "secret": sec}),
            credential_type="api_key",
            scopes=["accounts", "transactions", "liabilities", "investments"],
            metadata={"env": environment},
        )
        _log.info("plaid connected (env=%s)", environment)
        token = (access_token or "").strip() or self._env_token()
        if token:
            item = self.link_bank(token, institution_name=institution_name)
            return ConnectResult(
                ok=True,
                account=item["institution_name"],
                scopes=["accounts", "transactions", "liabilities",
                        "investments"],
                message=(
                    f"connected to Plaid ({environment}) and linked "
                    f"{item['institution_name']}. Credentials are in the "
                    "encrypted vault."
                ),
            )
        return ConnectResult(
            ok=True,
            account=f"plaid ({environment}, no banks linked)",
            scopes=["accounts", "transactions", "liabilities",
                    "investments"],
            message=(
                f"connected to Plaid ({environment}); client credentials "
                "validated and vault-stored. No bank linked yet — run "
                "begin_link() to start the Plaid Link flow, or "
                "link_bank(public_token=...) with a Link public_token."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        client = self._client_credential()
        if client is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name plaid`",
            )
        env = (client.metadata or {}).get("env", "sandbox")
        try:
            self._validate_client(
                BASE_URLS[env], *self._client_pair(client)
            )
        except PlaidError as exc:
            return ConnectorStatus(
                connected=False,
                account=f"plaid ({env})",
                scopes=list((client.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"client credentials rejected ({exc}): reconnect",
            )
        items = self._items()
        names = [i["institution_name"] for i in items]
        return ConnectorStatus(
            connected=True,
            account=", ".join(names) if names else f"plaid ({env}, no banks)",
            scopes=list((client.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"{len(items)} bank item(s) linked",
        )

    def test_connection(self) -> bool:
        client = self._client_credential()
        if client is None:
            return False
        env = (client.metadata or {}).get("env", "sandbox")
        try:
            self._validate_client(
                BASE_URLS[env], *self._client_pair(client)
            )
            return True
        except ConnectorError:
            return False

    # ── bank linking (human-in-the-loop) ─────────────────────────

    def create_link_token(
        self,
        *,
        products: list[str] | None = None,
        country_codes: list[str] | None = None,
        language: str = "en",
    ) -> dict[str, Any]:
        """Mint a Plaid Link token (``POST /link/token/create``).

        The owner completes this token in Plaid Link in their own browser;
        Link returns a ``public_token`` that :meth:`link_bank` exchanges.
        """
        client = self._require_client()
        env = (client.metadata or {}).get("env", "sandbox")
        cid, sec = self._client_pair(client)
        resp = self._api(
            BASE_URLS[env],
            "/link/token/create",
            {
                "client_id": cid,
                "secret": sec,
                "client_name": "Devon",
                "products": products or ["transactions"],
                "country_codes": country_codes or ["US"],
                "language": language,
                "user": {"client_user_id": "devon-owner"},
            },
            cid=cid,
            secret=sec,
        )
        link_token = str(resp.get("link_token", ""))
        if not link_token:
            raise PlaidError("plaid did not return a link_token")
        return {"link_token": link_token,
                "expiration": resp.get("expiration", "")}

    def link_page(self, link_token: str) -> str:
        """Write a one-page Plaid Link opener the owner opens in a browser.

        Returns the file path. The page initializes Plaid Link with the
        token; on success it displays the ``public_token`` for the owner to
        hand back to ``link_bank(public_token=...)``. Bank credentials are
        entered in Plaid's own iframe — Devon never sees them.
        """
        safe = link_token.replace("\\", "\\\\").replace('"', '\\"')
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Link a bank with Plaid</title>
<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
</head><body style="font-family:sans-serif;max-width:40em;margin:3em auto">
<h1>Link a bank account</h1>
<p>This page is served from your own machine by Devon. Sign in to your bank
inside Plaid's window — Devon never sees your bank credentials.</p>
<button id="link-btn" style="font-size:1.2em;padding:.6em 1.2em">Open Plaid Link</button>
<pre id="out" style="background:#f4f4f4;padding:1em;white-space:pre-wrap"></pre>
<script>
const out = document.getElementById('out');
const handler = Plaid.create({{
  token: "{safe}",
  onSuccess: (public_token, metadata) => {{
    out.textContent = "Linked " + metadata.institution.name + ".\\n\\n"
      + "Give this public_token to Devon:\\n" + public_token;
  }},
  onExit: (err, metadata) => {{
    out.textContent = "Link closed" + (err ? ": " + err.error_message : ".");
  }},
}});
document.getElementById('link-btn').onclick = () => handler.open();
</script></body></html>
"""
        path = Path(tempfile.mkstemp(prefix="devon-plaid-link-",
                                     suffix=".html")[1])
        path.write_text(html, encoding="utf-8")
        return str(path)

    def begin_link(
        self,
        *,
        products: list[str] | None = None,
        country_codes: list[str] | None = None,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Start the human bank-linking flow.

        Mints a Link token, writes the opener page, and — with ``db`` —
        pauses at a human checkpoint until the owner finishes Plaid Link in
        their browser and resolves the checkpoint with the public_token.
        Without ``db`` the guide is printed and the owner hands the
        public_token back to ``link_bank(public_token=...)``.
        """
        from .checkpoints import CheckpointKind, HumanCheckpointPending

        self._require_client()
        token = self.create_link_token(
            products=products, country_codes=country_codes
        )
        page = self.link_page(token["link_token"])
        guide = "\n".join([
            "Link a bank through Plaid (only you can do this):",
            f"1. Open this page in your browser: {page}",
            "2. Click 'Open Plaid Link', pick your bank, and sign in there.",
            "   Your bank credentials stay inside Plaid's window.",
            "3. On success the page shows a public_token — hand it to Devon:",
            "   link_bank(public_token=<public_token>).",
        ])
        print(guide)
        if db is None:
            return {
                "linked": False,
                "link_token": token["link_token"],
                "page": page,
                "next": "finish Plaid Link, then call "
                        "link_bank(public_token=<public_token>)",
            }
        try:
            self.request_human(
                CheckpointKind.MANUAL_STEP,
                "Link a bank account with Plaid",
                guide
                + "\n\nResolve this checkpoint with the public_token, e.g. "
                "note 'public_token=<token from the page>'.",
                db=db,
                context=context,
                resume_state={"stage": "link_bank"},
            )
        except HumanCheckpointPending as pending:
            return {
                "linked": False,
                "checkpoint_id": pending.checkpoint.id,
                "page": page,
                "next": (
                    "finish Plaid Link in the browser, then resolve "
                    f"checkpoint {pending.checkpoint.id} with the public_token"
                ),
            }
        raise ConnectorError(
            "linking confirmed interactively but no public_token was "
            "captured — call link_bank(public_token=<token from the page>)"
        )

    def link_bank(
        self, public_token: str, *, institution_name: str = ""
    ) -> dict[str, Any]:
        """Exchange a Link ``public_token`` for an access token and store it.

        ``POST /item/public_token/exchange``. The access token is
        vault-stored under the institution's name; the raw token is never
        returned to chat.
        """
        client = self._require_client()
        env = (client.metadata or {}).get("env", "sandbox")
        cid, sec = self._client_pair(client)
        token = (public_token or "").strip()
        if not token:
            raise ConnectorError(
                "empty public_token — finish Plaid Link first, then pass "
                "the token the page shows"
            )
        resp = self._api(
            BASE_URLS[env],
            "/item/public_token/exchange",
            {"client_id": cid, "secret": sec, "public_token": token},
            cid=cid,
            secret=sec,
        )
        access_token = str(resp.get("access_token", ""))
        item_id = str(resp.get("item_id", ""))
        if not access_token or not item_id:
            raise PlaidError(
                "plaid did not return an access_token/item_id for this "
                "public_token — it may have expired (exchange it promptly)"
            )
        accounts = self._api(
            BASE_URLS[env], "/accounts/get",
            {"client_id": cid, "secret": sec,
             "access_token": access_token},
            cid=cid, secret=sec,
        )
        item = accounts.get("item", {})
        name = (institution_name or "").strip() or self._institution_name(
            env, cid, sec, str(item.get("institution_id", ""))
        ) or f"item {item_id[-6:]}"
        self._store_credential(
            f"item:{item_id}",
            access_token,
            credential_type="access_token",
            scopes=["accounts", "transactions", "liabilities",
                    "investments"],
            metadata={
                "item_id": item_id,
                "institution_id": item.get("institution_id", ""),
                "institution_name": name,
                "cursor": None,
            },
        )
        _log.info("plaid bank linked: %s", name)
        return {
            "item_id": item_id,
            "institution_name": name,
            "accounts": len(accounts.get("accounts", [])),
        }

    def remove_item(
        self, *, access_token: str = "", item_id: str = ""
    ) -> dict[str, Any]:
        """Remove one linked bank (``POST /item/remove``), server-side too.

        This invalidates the access token at Plaid, then drops the local
        record. (Plaid's own guidance: per-bank removal from chat isn't
        offered — this is the API path for an explicit owner request.)
        """
        client = self._require_client()
        items = self._items(access_token=access_token or None)
        target = None
        for item in items:
            if (item_id and item["item_id"] == item_id) or (
                access_token and item["access_token"] == access_token
            ):
                target = item
                break
        if target is None:
            raise ConnectorError(
                "no linked bank matches — see status() for linked banks"
            )
        env = (client.metadata or {}).get("env", "sandbox")
        cid, sec = self._client_pair(client)
        self._api(
            BASE_URLS[env],
            "/item/remove",
            {"client_id": cid, "secret": sec,
             "access_token": target["access_token"]},
            cid=cid,
            secret=sec,
        )
        self.vault.delete(self._service, f"item:{target['item_id']}")
        _log.info("plaid item removed: %s", target["institution_name"])
        return {"removed": True,
                "institution_name": target["institution_name"]}

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue after the owner resolved a human checkpoint.

        Dispatches on the checkpoint's ``resume_state``: the Plaid Link
        flow (``stage == "link_bank"``) or a money-movement approval
        (``intent`` of ``create_transfer`` / ``cancel_transfer`` /
        ``create_payment`` / ``reverse_payment``).
        """
        from .checkpoints import CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must act first"
            )
        state = checkpoint.resume_state or {}
        if state.get("stage") == "link_bank":
            public_token = self._token_from_note(checkpoint.result_note or "")
            if not public_token:
                raise ConnectorError(
                    "the resolved checkpoint has no public_token — resolve "
                    "it again with note 'public_token=<token from the page>'"
                )
            return self.link_bank(public_token)
        intent = state.get("intent", "")
        if intent == "create_transfer":
            return self.create_transfer(
                access_token=state["access_token"],
                account_id=state["account_id"],
                authorization_id=state["authorization_id"],
                type=state["type"],
                network=state.get("network", "ach"),
                amount=state["amount"],
                description=state.get("description", ""),
                ach_class=state.get("ach_class", "ppd"),
                legal_name=state.get("legal_name", ""),
                idempotency_key=state.get("idempotency_key"),
            )
        if intent == "cancel_transfer":
            return self.cancel_transfer(
                state["transfer_id"],
                access_token=state["access_token"],
            )
        if intent == "create_payment":
            return self.create_payment(
                state["recipient_id"],
                state["reference"],
                state["amount"],
                currency=state.get("currency", "GBP"),
                user_id=state.get("user_id", ""),
                idempotency_key=state.get("idempotency_key"),
            )
        if intent == "reverse_payment":
            return self.reverse_payment(state["payment_id"])
        raise ConnectorError(
            "plaid cannot resume checkpoint "
            f"{checkpoint.id}: unknown intent {intent!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def get_accounts(
        self, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Accounts across linked banks (``POST /accounts/get``).

        This is also the balance read — each account carries ``balances``
        (``available``/``current``/``limit``). Plaid figures are the last
        reported, not live.
        """
        out = []
        for item in self._items(access_token=access_token):
            resp = self._authed(item, "/accounts/get", {})
            out.append({
                "institution_name": item["institution_name"],
                "item_id": item["item_id"],
                "accounts": resp.get("accounts", []),
            })
        return out

    def get_balances(
        self, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Balances per account (derived from ``/accounts/get``)."""
        out = []
        for entry in self.get_accounts(access_token=access_token):
            accounts = []
            for acct in entry["accounts"]:
                balances = acct.get("balances", {}) or {}
                accounts.append({
                    "name": acct.get("name", ""),
                    "mask": acct.get("mask", ""),
                    "type": acct.get("type", ""),
                    "subtype": acct.get("subtype", ""),
                    "currency": balances.get("iso_currency_code", ""),
                    "available": balances.get("available"),
                    "current": balances.get("current"),
                    "limit": balances.get("limit"),
                })
            out.append({
                "institution_name": entry["institution_name"],
                "accounts": accounts,
            })
        return out

    def get_transactions(
        self,
        start_date: str,
        end_date: str,
        *,
        access_token: str | None = None,
        account_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Transaction history (``POST /transactions/get``).

        Dates are inclusive ``YYYY-MM-DD``. Pages automatically until
        ``has_more`` is false. Plaid convention: positive amounts are
        outflows, negative amounts are inflows.
        """
        self._check_dates(start_date, end_date)
        out = []
        for item in self._items(access_token=access_token):
            options: dict[str, Any] = {}
            if account_ids:
                options["account_ids"] = list(account_ids)
            txns: list[dict[str, Any]] = []
            offset = 0
            while True:
                resp = self._authed(item, "/transactions/get", {
                    "access_token": item["access_token"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "options": {**options, "count": 100, "offset": offset},
                })
                txns.extend(resp.get("transactions", []))
                if not resp.get("has_more"):
                    break
                offset += 100
            out.append({
                "institution_name": item["institution_name"],
                "accounts": resp.get("accounts", []),
                "transactions": txns,
                "total": resp.get("total_transactions"),
            })
        return out

    def sync_transactions(
        self,
        *,
        access_token: str | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        """Incremental sync (``POST /transactions/sync``).

        For persistent stores: applies pages until ``has_more`` is false and
        persists the returned cursor in the vault per item. Pass ``cursor``
        to override the stored one. For one-off date-range reports without
        a store, use :meth:`get_transactions` instead.
        """
        out = []
        for item in self._items(access_token=access_token):
            cur = cursor if cursor is not None else item["metadata"].get(
                "cursor")
            added: list[dict[str, Any]] = []
            modified: list[dict[str, Any]] = []
            removed: list[dict[str, Any]] = []
            while True:
                body: dict[str, Any] = {
                    "access_token": item["access_token"],
                    "count": 100,
                }
                if cur:
                    body["cursor"] = cur
                resp = self._authed(item, "/transactions/sync", body)
                added.extend(resp.get("added", []))
                modified.extend(resp.get("modified", []))
                removed.extend(resp.get("removed", []))
                cur = resp.get("next_cursor", cur)
                if not resp.get("has_more"):
                    break
            self._save_cursor(item, cur)
            out.append({
                "institution_name": item["institution_name"],
                "added": added,
                "modified": modified,
                "removed": removed,
                "cursor": cur,
                "has_more": False,
            })
        return out

    def get_recurring(
        self,
        *,
        access_token: str | None = None,
        account_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recurring streams (``POST /transactions/recurring/get``).

        ``inflow_streams`` (e.g. paychecks) and ``outflow_streams`` (e.g.
        subscriptions). ``is_active`` marks patterns Plaid still considers
        ongoing.
        """
        out = []
        for item in self._items(access_token=access_token):
            body: dict[str, Any] = {"access_token": item["access_token"]}
            if account_ids:
                body["account_ids"] = list(account_ids)
            resp = self._authed(item, "/transactions/recurring/get", body)
            out.append({
                "institution_name": item["institution_name"],
                "inflow_streams": resp.get("inflow_streams", []),
                "outflow_streams": resp.get("outflow_streams", []),
            })
        return out

    def get_liabilities(
        self, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Loans and credit (``POST /liabilities/get``): balances, rates,
        minimum payments, due dates."""
        out = []
        for item in self._items(access_token=access_token):
            resp = self._authed(item, "/liabilities/get", {})
            out.append({
                "institution_name": item["institution_name"],
                "liabilities": resp.get("liabilities", {}),
                "accounts": resp.get("accounts", []),
            })
        return out

    def get_investment_holdings(
        self, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Investment positions (``POST /investments/holdings/get``)."""
        out = []
        for item in self._items(access_token=access_token):
            resp = self._authed(item, "/investments/holdings/get", {})
            out.append({
                "institution_name": item["institution_name"],
                "accounts": resp.get("accounts", []),
                "holdings": resp.get("holdings", []),
                "securities": resp.get("securities", []),
            })
        return out

    def get_investment_transactions(
        self,
        start_date: str,
        end_date: str,
        *,
        access_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Investment activity (``POST /investments/transactions/get``).

        Pages with offset/count until the fetched count reaches
        ``total_investment_transactions``.
        """
        self._check_dates(start_date, end_date)
        out = []
        for item in self._items(access_token=access_token):
            txns: list[dict[str, Any]] = []
            offset = 0
            total = None
            while True:
                resp = self._authed(item, "/investments/transactions/get", {
                    "access_token": item["access_token"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "options": {"count": 100, "offset": offset},
                })
                txns.extend(resp.get("investment_transactions", []))
                total = resp.get("total_investment_transactions", total)
                offset += len(resp.get("investment_transactions", []))
                if total is not None and offset >= total:
                    break
                if not resp.get("investment_transactions"):
                    break
            out.append({
                "institution_name": item["institution_name"],
                "investment_transactions": txns,
                "total": total,
            })
        return out

    # ── transfers (US ACH) ───────────────────────────────────────

    def create_transfer_authorization(
        self,
        *,
        access_token: str | None = None,
        account_id: str = "",
        type: str = "debit",
        network: str = "ach",
        amount: str = "",
        ach_class: str = "ppd",
        legal_name: str = "",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Authorize a transfer (``POST /transfer/authorization/create``).

        Returns an ``authorization_id`` — pass it to :meth:`create_transfer`.
        The authorization_id doubles as Plaid's idempotency key: creating
        twice with the same id returns the same transfer, never a duplicate
        (the standalone ``idempotency_key`` request field is deprecated per
        the current API reference). ``idempotency_key`` here is the
        *authorization* idempotency key (Plaid expires it after 48h —
        pass your own for safe retries of this call).

        ``amount`` is the MAXIMUM as a decimal string (``"25.00"``);
        ``type`` is ``"debit"`` (pull from the account) or ``"credit"``
        (push to it). Raises a clear error if Transfer isn't enabled on
        the Plaid account.
        """
        item = self._single_item(access_token)
        if type not in ("debit", "credit"):
            raise ConnectorError(
                f"invalid transfer type {type!r}: use 'debit' or 'credit'"
            )
        self._check_amount_str(amount, "transfer authorization")
        body: dict[str, Any] = {
            "access_token": item["access_token"],
            "account_id": account_id,
            "type": type,
            "network": network,
            "amount": amount,
            "ach_class": ach_class,
            "user": {"legal_name": legal_name or "Devon Owner"},
        }
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        try:
            resp = self._authed(item, "/transfer/authorization/create", body)
        except PlaidError as exc:
            raise self._transfer_not_enabled(exc) from exc
        return {
            "authorization_id": resp.get("authorization_id", ""),
            "account_id": account_id,
            "type": type,
            "amount": amount,
        }

    def create_transfer(
        self,
        *,
        access_token: str | None = None,
        account_id: str = "",
        authorization_id: str = "",
        type: str = "debit",
        network: str = "ach",
        amount: str = "",
        description: str = "",
        ach_class: str = "ppd",
        legal_name: str = "",
        confirm: bool = False,
        db: Any = None,
        context: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create a US ACH transfer (``POST /transfer/create``).

        ``confirm=True`` (with ``db=``) pauses at a human checkpoint showing
        the exact amount, direction, and account — the owner approves before
        money moves. Without ``confirm`` the transfer executes immediately
        (for callers that already secured approval).

        Idempotency: ``idempotency_key`` (generated if omitted) is checked
        against the audit log first — a replayed key returns the original
        result without a new API call. Plaid's own idempotency rides on the
        ``authorization_id``.
        """
        item = self._single_item(access_token)
        if type not in ("debit", "credit"):
            raise ConnectorError(
                f"invalid transfer type {type!r}: use 'debit' or 'credit'"
            )
        self._check_amount_str(amount, "transfer")
        if not authorization_id.strip():
            raise ConnectorError(
                "authorization_id is required — call "
                "create_transfer_authorization first"
            )
        key = idempotency_key or self._new_idempotency_key()
        prior = self._audit_find(key)
        if prior is not None:
            return {
                "transfer_id": prior.get("transfer_id", ""),
                "status": prior.get("status", ""),
                "idempotent_replay": True,
                "idempotency_key": key,
            }
        direction = "from" if type == "debit" else "to"
        if confirm:
            if db is None:
                raise ConnectorError(
                    "transfer confirmation needs a database for "
                    "checkpoints (pass db=)"
                )
            from .checkpoints import CheckpointKind

            cp = self.request_human(
                CheckpointKind.MANUAL_STEP,
                f"Approve ACH transfer — {amount} USD {direction} account",
                "\n".join([
                    "Devon is about to move real money via Plaid Transfer:",
                    f"  direction: {type} ({direction} the linked account)",
                    f"  amount:    USD {amount}",
                    f"  account:   {account_id}",
                    f"  reference: {description or '(none)'}",
                    "Nothing moves until you approve.",
                ]),
                db=db,
                context=context,
                resume_state={
                    "intent": "create_transfer",
                    "access_token": item["access_token"],
                    "account_id": account_id,
                    "authorization_id": authorization_id,
                    "type": type,
                    "network": network,
                    "amount": amount,
                    "description": description,
                    "ach_class": ach_class,
                    "legal_name": legal_name,
                    "idempotency_key": key,
                },
            )
            return self.resume_checkpoint(cp, db=db, context=context)
        body: dict[str, Any] = {
            "access_token": item["access_token"],
            "account_id": account_id,
            "authorization_id": authorization_id,
            "type": type,
            "network": network,
            "amount": amount,
            "description": (description or "Devon transfer")[:10],
            "ach_class": ach_class,
            "user": {"legal_name": legal_name or "Devon Owner"},
        }
        try:
            resp = self._authed(item, "/transfer/create", body)
        except PlaidError as exc:
            raise self._transfer_not_enabled(exc) from exc
        transfer_id = str(resp.get("transfer_id", ""))
        result = {
            "transfer_id": transfer_id,
            "status": resp.get("status", ""),
            "idempotency_key": key,
        }
        self._audit("transfer_create", {
            "idempotency_key": key,
            "transfer_id": transfer_id,
            "status": result["status"],
            "type": type,
            "amount": amount,
            "currency": "USD",
            "account_id": account_id,
            "description": description,
        })
        _log.info("plaid transfer created: %s (%s USD %s)",
                  transfer_id, amount, type)
        return result

    def get_transfer(
        self, transfer_id: str, *, access_token: str | None = None
    ) -> dict[str, Any]:
        """Transfer status and detail (``POST /transfer/get``)."""
        item = self._single_item(access_token)
        if not transfer_id.strip():
            raise ConnectorError("transfer_id is required")
        resp = self._authed(
            item, "/transfer/get", {"transfer_id": transfer_id})
        return resp.get("transfer", resp)

    def list_transfers(
        self,
        *,
        access_token: str | None = None,
        count: int = 25,
    ) -> list[dict[str, Any]]:
        """Recent transfers (``POST /transfer/list``)."""
        item = self._single_item(access_token)
        resp = self._authed(
            item, "/transfer/list", {"count": max(1, min(count, 100))})
        return resp.get("transfers", [])

    def cancel_transfer(
        self,
        transfer_id: str,
        *,
        access_token: str | None = None,
        confirm: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Cancel a pending transfer (``POST /transfer/cancel``).

        Only pending transfers can be cancelled. ``confirm=True`` pauses
        at a human checkpoint first.
        """
        item = self._single_item(access_token)
        if not transfer_id.strip():
            raise ConnectorError("transfer_id is required")
        if confirm:
            if db is None:
                raise ConnectorError(
                    "cancel confirmation needs a database for "
                    "checkpoints (pass db=)"
                )
            from .checkpoints import CheckpointKind

            cp = self.request_human(
                CheckpointKind.MANUAL_STEP,
                f"Approve transfer cancellation — {transfer_id}",
                "\n".join([
                    "Devon is about to cancel this Plaid transfer:",
                    f"  transfer_id: {transfer_id}",
                    "Only pending transfers can be cancelled.",
                ]),
                db=db,
                context=context,
                resume_state={
                    "intent": "cancel_transfer",
                    "access_token": item["access_token"],
                    "transfer_id": transfer_id,
                },
            )
            return self.resume_checkpoint(cp, db=db, context=context)
        resp = self._authed(
            item, "/transfer/cancel", {"transfer_id": transfer_id})
        self._audit("transfer_cancel", {"transfer_id": transfer_id})
        return {"cancelled": True, "transfer_id": transfer_id,
                "status": resp.get("status", "")}

    def list_transfer_events(
        self,
        *,
        access_token: str | None = None,
        transfer_id: str = "",
        count: int = 25,
    ) -> list[dict[str, Any]]:
        """Transfer lifecycle events (``POST /transfer/event/list``).

        Posted/settled/returned/failed transitions land here — poll this
        (or wire the TRANSFER_EVENTS_UPDATE webhook) to track a transfer.
        """
        item = self._single_item(access_token)
        body: dict[str, Any] = {"count": max(1, min(count, 100))}
        if transfer_id.strip():
            body["transfer_id"] = transfer_id
        resp = self._authed(item, "/transfer/event/list", body)
        return resp.get("transfer_events", [])

    # ── payment initiation (UK/EU) ───────────────────────────────

    def create_payment_recipient(
        self,
        name: str,
        *,
        iban: str = "",
        bacs_account: str = "",
        bacs_sort_code: str = "",
        address: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a payment recipient
        (``POST /payment_initiation/recipient/create``).

        UK: pass ``bacs_account`` + ``bacs_sort_code`` (iban/address not
        needed). EU: pass ``iban`` + ``address`` (street/city/postal_code/
        country). Returns a ``recipient_id`` for :meth:`create_payment`.
        """
        self._require_client()
        if not name.strip():
            raise ConnectorError("recipient name is required")
        body: dict[str, Any] = {"name": name.strip()}
        if bacs_account and bacs_sort_code:
            body["bacs"] = {"account": bacs_account,
                            "sort_code": bacs_sort_code}
        elif iban.strip():
            body["iban"] = iban.strip()
            if address:
                body["address"] = dict(address)
        else:
            raise ConnectorError(
                "recipient needs bacs (UK) or iban (EU) details"
            )
        resp = self._client_api("/payment_initiation/recipient/create", body)
        return {
            "recipient_id": resp.get("recipient_id", ""),
            "request_id": resp.get("request_id", ""),
        }

    def get_payment_recipient(
        self, recipient_id: str
    ) -> dict[str, Any]:
        """Fetch a recipient (``POST /payment_initiation/recipient/get``)."""
        self._require_client()
        if not recipient_id.strip():
            raise ConnectorError("recipient_id is required")
        return self._client_api(
            "/payment_initiation/recipient/get",
            {"recipient_id": recipient_id},
        )

    def list_payment_recipients(self) -> list[dict[str, Any]]:
        """All recipients (``POST /payment_initiation/recipient/list``)."""
        self._require_client()
        resp = self._client_api(
            "/payment_initiation/recipient/list", {})
        return resp.get("recipients", [])

    def create_payment_user(
        self,
        name: str,
        *,
        email: str = "",
        phone: str = "",
    ) -> dict[str, Any]:
        """Create a Plaid end user (``POST /user/create``).

        New Plaid integrations MUST pass a ``user_id`` when creating a
        payment (``/payment_initiation/payment/create``) — create the user
        first with this method and hand the returned ``user_id`` to
        :meth:`create_payment`. The user needs a name plus an email
        address or a phone number (per the Plaid API reference).
        """
        self._require_client()
        if not (name or "").strip():
            raise ConnectorError("payment user name is required")
        if not (email or "").strip() and not (phone or "").strip():
            raise ConnectorError(
                "a payment user needs an email address or a phone number"
            )
        body: dict[str, Any] = {"client_user_id": name.strip()[:64]}
        if email.strip():
            body["email_address"] = email.strip()
        if phone.strip():
            body["phone_number"] = phone.strip()
        resp = self._client_api("/user/create", body)
        user_id = str(resp.get("user_id", ""))
        if not user_id:
            raise PlaidError("plaid did not return a user_id")
        return {"user_id": user_id, "request_id": resp.get("request_id", "")}

    def create_payment(
        self,
        recipient_id: str,
        reference: str,
        amount: float,
        *,
        currency: str = "GBP",
        user_id: str = "",
        confirm: bool = False,
        db: Any = None,
        context: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create a UK/EU payment (``POST /payment_initiation/payment/create``).

        Returns ``payment_id`` + ``status``. The owner then authorizes the
        payment in Plaid Link (mint a link token with
        ``products=["payment_initiation"]`` and the payment_id).

        ``user_id`` (from :meth:`create_payment_user`) is REQUIRED for new
        Plaid integrations per the current API reference; older
        integrations may omit it. ``reference`` must be alphanumeric,
        1–18 chars (bank rule — Plaid truncates overlong ones itself).

        ``confirm=True`` (with ``db=``) pauses at a human checkpoint showing
        the exact amount, currency, and recipient — the owner approves
        before the payment is created. Idempotency-keyed and audit-logged
        like transfers.
        """
        self._require_client()
        if not recipient_id.strip():
            raise ConnectorError("recipient_id is required")
        reference = (reference or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9]{1,18}", reference):
            raise ConnectorError(
                f"invalid payment reference {reference!r}: must be "
                "alphanumeric, 1-18 characters (bank rule)"
            )
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ConnectorError(
                f"invalid payment amount {amount!r}: must be positive"
            )
        currency = (currency or "GBP").upper()
        key = idempotency_key or self._new_idempotency_key()
        prior = self._audit_find(key)
        if prior is not None:
            return {
                "payment_id": prior.get("payment_id", ""),
                "status": prior.get("status", ""),
                "idempotent_replay": True,
                "idempotency_key": key,
            }
        if confirm:
            if db is None:
                raise ConnectorError(
                    "payment confirmation needs a database for "
                    "checkpoints (pass db=)"
                )
            from .checkpoints import CheckpointKind

            cp = self.request_human(
                CheckpointKind.MANUAL_STEP,
                f"Approve payment — {currency} {amount:.2f} to recipient",
                "\n".join([
                    "Devon is about to create a Plaid payment:",
                    f"  amount:    {currency} {amount:.2f}",
                    f"  recipient: {recipient_id}",
                    f"  reference: {reference}",
                    "You will then authorize it in Plaid Link.",
                    "Nothing moves until you approve.",
                ]),
                db=db,
                context=context,
                resume_state={
                    "intent": "create_payment",
                    "recipient_id": recipient_id,
                    "reference": reference,
                    "amount": amount,
                    "currency": currency,
                    "user_id": user_id,
                    "idempotency_key": key,
                },
            )
            return self.resume_checkpoint(cp, db=db, context=context)
        payload: dict[str, Any] = {
            "recipient_id": recipient_id,
            "reference": reference,
            "amount": {"currency": currency, "value": amount},
        }
        if user_id.strip():
            payload["user_id"] = user_id.strip()
        resp = self._client_api(
            "/payment_initiation/payment/create",
            payload,
        )
        payment_id = str(resp.get("payment_id", ""))
        result = {
            "payment_id": payment_id,
            "status": resp.get("status", ""),
            "idempotency_key": key,
        }
        self._audit("payment_create", {
            "idempotency_key": key,
            "payment_id": payment_id,
            "status": result["status"],
            "amount": amount,
            "currency": currency,
            "recipient_id": recipient_id,
            "reference": reference,
        })
        _log.info("plaid payment created: %s (%s %.2f)",
                  payment_id, currency, amount)
        return result

    def get_payment(self, payment_id: str) -> dict[str, Any]:
        """Payment status (``POST /payment_initiation/payment/get``)."""
        self._require_client()
        if not payment_id.strip():
            raise ConnectorError("payment_id is required")
        return self._client_api(
            "/payment_initiation/payment/get",
            {"payment_id": payment_id},
        )

    def list_payments(self, count: int = 25) -> list[dict[str, Any]]:
        """Recent payments (``POST /payment_initiation/payment/list``)."""
        self._require_client()
        resp = self._client_api(
            "/payment_initiation/payment/list",
            {"count": max(1, min(count, 100))},
        )
        return resp.get("payments", [])

    def reverse_payment(
        self,
        payment_id: str,
        *,
        confirm: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Refund a payment from a virtual account
        (``POST /payment_initiation/payment/reverse``).

        ``confirm=True`` pauses at a human checkpoint first.
        """
        self._require_client()
        if not payment_id.strip():
            raise ConnectorError("payment_id is required")
        if confirm:
            if db is None:
                raise ConnectorError(
                    "reversal confirmation needs a database for "
                    "checkpoints (pass db=)"
                )
            from .checkpoints import CheckpointKind

            cp = self.request_human(
                CheckpointKind.MANUAL_STEP,
                f"Approve payment reversal — {payment_id}",
                "\n".join([
                    "Devon is about to reverse this Plaid payment:",
                    f"  payment_id: {payment_id}",
                    "Nothing moves until you approve.",
                ]),
                db=db,
                context=context,
                resume_state={
                    "intent": "reverse_payment",
                    "payment_id": payment_id,
                },
            )
            return self.resume_checkpoint(cp, db=db, context=context)
        resp = self._client_api(
            "/payment_initiation/payment/reverse",
            {"payment_id": payment_id},
        )
        self._audit("payment_reverse", {"payment_id": payment_id})
        return {"reversed": True, "payment_id": payment_id,
                "status": resp.get("status", "")}

    def sandbox_public_token(
        self, institution_id: str = "ins_109508"
    ) -> dict[str, Any]:
        """Create a sandbox public_token for testing (sandbox env only).

        ``POST /sandbox/public_token/create``. Exchange it with
        :meth:`link_bank`. Never available in production.
        """
        client = self._require_client()
        env = (client.metadata or {}).get("env", "sandbox")
        if env != "sandbox":
            raise ConnectorError(
                "sandbox public tokens are only available in the sandbox "
                "environment"
            )
        cid, sec = self._client_pair(client)
        resp = self._api(
            BASE_URLS[env],
            "/sandbox/public_token/create",
            {
                "client_id": cid,
                "secret": sec,
                "institution_id": institution_id,
                "initial_products": ["transactions"],
            },
            cid=cid,
            secret=sec,
        )
        return {
            "public_token": resp.get("public_token", ""),
            "request_id": resp.get("request_id", ""),
        }

    # ── owner-safe summaries ─────────────────────────────────────

    def summarize_accounts(
        self, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Accounts in plain words: names, types, masked digits, balances.

        No internal ids, no cursors — safe to relay to the owner.
        """
        out = []
        for entry in self.get_balances(access_token=access_token):
            accounts = []
            for acct in entry["accounts"]:
                mask = acct.get("mask") or ""
                label = acct["name"]
                if mask:
                    label += f" (…{mask})"
                accounts.append({
                    "account": label,
                    "type": self._pretty_type(acct.get("type", ""),
                                             acct.get("subtype", "")),
                    "currency": acct.get("currency", ""),
                    "available": acct.get("available"),
                    "current": acct.get("current"),
                    "limit": acct.get("limit"),
                })
            out.append({
                "institution": entry["institution_name"],
                "note": "balances are Plaid's last reported figures, "
                        "not live",
                "accounts": accounts,
            })
        return out

    def summarize_transactions(
        self,
        start_date: str,
        end_date: str,
        *,
        access_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Transactions in plain words: date, merchant, amount, direction.

        Plaid's sign convention is kept: positive amounts are money out,
        negative amounts are money in (refunds included).
        """
        out = []
        for entry in self.get_transactions(
            start_date, end_date, access_token=access_token
        ):
            names = {
                a.get("account_id"): a.get("name", "")
                for a in entry["accounts"]
            }
            txns = []
            for txn in entry["transactions"]:
                merchant = (
                    txn.get("merchant_name") or txn.get("name") or ""
                )
                txns.append({
                    "date": txn.get("date", ""),
                    "description": merchant,
                    "amount": txn.get("amount"),
                    "currency": txn.get("iso_currency_code", ""),
                    "direction": "out"
                    if (txn.get("amount") or 0) > 0 else "in",
                    "pending": bool(txn.get("pending")),
                    "account": names.get(txn.get("account_id", ""), ""),
                })
            out.append({
                "institution": entry["institution_name"],
                "range": f"{start_date} to {end_date} "
                         "(Plaid posting dates; data may lag the bank)",
                "transactions": txns,
            })
        return out

    def summarize_recurring(
        self, *, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Active recurring money in/out in plain words."""
        out = []
        for entry in self.get_recurring(access_token=access_token):
            def _clean(streams: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [{
                    "description": s.get("description", ""),
                    "average_amount": s.get("average_amount"),
                    "currency": s.get("iso_currency_code", ""),
                    "frequency": s.get("frequency", ""),
                    "last_date": s.get("last_date", ""),
                    "is_active": bool(s.get("is_active")),
                } for s in streams if s.get("is_active")]
            out.append({
                "institution": entry["institution_name"],
                "note": "average amounts can include one-off payments; "
                        "check recent transactions before quoting a cost",
                "money_in": _clean(entry["inflow_streams"]),
                "money_out": _clean(entry["outflow_streams"]),
            })
        return out

    def summarize_liabilities(
        self, *, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """What is owed, in plain words: balances, rates, due dates."""
        out = []
        for entry in self.get_liabilities(access_token=access_token):
            liab = entry["liabilities"] or {}
            cards = []
            for card in (liab.get("credit") or []):
                cards.append({
                    "last_statement_balance":
                        card.get("last_statement_balance"),
                    "last_statement_issue_date":
                        card.get("last_statement_issue_date", ""),
                    "minimum_payment_amount":
                        card.get("minimum_payment_amount"),
                    "next_payment_due_date":
                        card.get("next_payment_due_date", ""),
                    "is_overdue": card.get("is_overdue"),
                })
            out.append({
                "institution": entry["institution_name"],
                "credit_cards": cards,
                "mortgages": liab.get("mortgage") or [],
                "student_loans": liab.get("student") or [],
                "auto_loans": liab.get("auto") or [],
            })
        return out

    # ── HTTP plumbing ────────────────────────────────────────────

    def _api(
        self,
        base: str,
        path: str,
        payload: dict[str, Any],
        *,
        cid: str,
        secret: str,
    ) -> dict[str, Any]:
        """One Plaid API call; errors become PlaidError with error_code."""
        url = f"{base}{path}"
        body = {"client_id": cid, "secret": secret, **payload}
        try:
            resp = self.http.post_json(
                url, body, headers={"Content-Type": "application/json"}
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise PlaidError(f"plaid request failed: {exc}") from exc
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001 - error bodies may not be JSON
            data = {}
        if not isinstance(data, dict):
            data = {}
        if not resp.ok or data.get("error_code"):
            code = str(data.get("error_code", ""))
            guidance = _ERROR_GUIDANCE.get(code, "")
            detail = str(
                data.get("display_message")
                or data.get("error_message")
                or resp.text[:200]
            )
            raise PlaidError(
                f"plaid {path} failed ({code or resp.status}): {detail}"
                + (f" — {guidance}" if guidance else ""),
                status_code=resp.status,
                error_code=code,
                error_type=str(data.get("error_type", "")),
            )
        return data

    def _authed(
        self, item: dict[str, Any], path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Plaid call for one linked item (client pair injected)."""
        client = self._require_client()
        env = (client.metadata or {}).get("env", "sandbox")
        cid, sec = self._client_pair(client)
        merged = {"access_token": item["access_token"]}
        merged.update(payload)
        return self._api(BASE_URLS[env], path, merged, cid=cid, secret=sec)

    # ── credential helpers ───────────────────────────────────────

    def _client_credential(self):
        try:
            return self.vault.get(self._service, "__client__")
        except Exception:  # noqa: BLE001 - absent credential is normal
            return None

    def _require_client(self):
        client = self._client_credential()
        if client is None:
            raise ConnectorError(
                "plaid is not connected — run "
                "`nm connectors connect --name plaid` first"
            )
        return client

    @staticmethod
    def _client_pair(client: Any) -> tuple[str, str]:
        try:
            pair = json.loads(client.password or "{}")
            return str(pair["client_id"]), str(pair["secret"])
        except (ValueError, KeyError) as exc:
            raise ConnectorError(
                "stored plaid client credential is corrupt — "
                "disconnect and reconnect"
            ) from exc

    def _items(
        self, access_token: str | None = None
    ) -> list[dict[str, Any]]:
        """Linked bank items; or a single ad-hoc item when access_token given."""
        if access_token:
            return [{
                "access_token": access_token,
                "item_id": "",
                "institution_name": "bank",
                "metadata": {},
                "_ephemeral": True,
            }]
        items = []
        for cred in self.vault.list_all(service=self._service):
            if cred.username == "__client__":
                continue
            try:
                full = self.vault.get(self._service, cred.username)
            except Exception:  # noqa: BLE001 - skip unreadable entries
                continue
            meta = full.metadata or {}
            items.append({
                "access_token": full.password,
                "item_id": meta.get("item_id", ""),
                "institution_name": meta.get("institution_name", "bank"),
                "metadata": meta,
                "_ephemeral": False,
            })
        if not items:
            raise ConnectorError(
                "no banks linked yet — run begin_link() to link one "
                "through Plaid Link"
            )
        return items

    def _save_cursor(self, item: dict[str, Any], cursor: str | None) -> None:
        if item.get("_ephemeral") or not item.get("item_id"):
            return
        meta = dict(item["metadata"] or {})
        meta["cursor"] = cursor
        self._store_credential(
            f"item:{item['item_id']}",
            item["access_token"],
            credential_type="access_token",
            scopes=["accounts", "transactions", "liabilities",
                    "investments"],
            metadata=meta,
        )

    def _institution_name(
        self, env: str, cid: str, sec: str, institution_id: str
    ) -> str:
        if not institution_id:
            return ""
        try:
            resp = self._api(
                env,
                "/institutions/get_by_id",
                {"institution_id": institution_id,
                 "country_codes": ["US"]},
                cid=cid,
                secret=sec,
            )
            return str((resp.get("institution") or {}).get("name", ""))
        except PlaidError:
            return ""

    # ── money-movement helpers ─────────────────────────────────

    def _single_item(self, access_token: str | None) -> dict[str, Any]:
        """One linked bank item for money movement.

        Transfers need a concrete account, so unlike the read methods
        (which fan out across banks), money movement requires exactly one
        item: pass ``access_token`` explicitly or have exactly one bank
        linked.
        """
        items = self._items(access_token=access_token)
        if len(items) > 1 and access_token is None:
            raise ConnectorError(
                f"{len(items)} banks linked — pass access_token to pick "
                "which account the money moves through"
            )
        return items[0]

    def _client_api(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Plaid call with client credentials only (no access token).

        Used by Payment Initiation, which is client-scoped.
        """
        client = self._require_client()
        env = (client.metadata or {}).get("env", "sandbox")
        cid, sec = self._client_pair(client)
        return self._api(BASE_URLS[env], path, payload, cid=cid, secret=sec)

    @staticmethod
    def _check_amount_str(amount: str, label: str) -> None:
        """Validate a Plaid decimal-string amount like ``"25.00"``."""
        if not re.match(r"^\d+\.\d{2}$", (amount or "").strip()):
            raise ConnectorError(
                f"invalid {label} amount {amount!r}: use a decimal string "
                'like "25.00"'
            )
        if float(amount) <= 0:
            raise ConnectorError(
                f"invalid {label} amount {amount!r}: must be positive"
            )

    @staticmethod
    def _transfer_not_enabled(exc: PlaidError) -> PlaidError:
        """Give a clear error when Transfer isn't enabled on the account."""
        code = (exc.error_code or "").upper()
        if code in ("TRANSFER_NOT_ENABLED", "PRODUCT_NOT_ENABLED",
                    "INVALID_PRODUCT"):
            return PlaidError(
                "Plaid Transfer is not enabled on this account — enable it "
                "in the Plaid dashboard (Transfer product) before creating "
                f"transfers (plaid said: {exc.error_code})",
                status_code=exc.status_code,
                error_code=exc.error_code,
                error_type=exc.error_type,
            )
        return exc

    @staticmethod
    def _new_idempotency_key() -> str:
        import uuid

        return f"devon-{uuid.uuid4().hex[:16]}"

    def _audit_path(self) -> Path:
        """JSONL audit log for every money movement."""
        base = Path(os.environ.get(
            "DEVON_AUDIT_DIR",
            str(Path.home() / ".devon" / "audit"),
        ))
        base.mkdir(parents=True, exist_ok=True)
        return base / "plaid-money.jsonl"

    def _audit(self, action: str, details: dict[str, Any]) -> None:
        """Append one money-movement record to the audit log."""
        record = {
            "ts": time.time(),
            "action": action,
            **details,
        }
        path = self._audit_path()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _audit_find(self, idempotency_key: str) -> dict[str, Any] | None:
        """Find a prior money movement by idempotency key, if any."""
        path = self._audit_path()
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("idempotency_key") == idempotency_key:
                        return record
        except OSError:
            return None
        return None

    def audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent money movements from the audit log (newest last)."""
        path = self._audit_path()
        if not path.exists():
            return []
        records = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except ValueError:
                            continue
        except OSError:
            return []
        return records[-max(1, limit):]

    # ── misc helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize_env(env: str | None) -> str:
        value = ((env or "").strip() or os.environ.get(ENV_ENV, "")
                 or "sandbox")
        value = value.lower()
        if value not in BASE_URLS:
            raise ConnectorError(
                f"invalid plaid env {value!r}: use 'sandbox' or 'production'"
            )
        return value

    @staticmethod
    def _env_token() -> str:
        return os.environ.get(ACCESS_TOKEN_ENV, "").strip()

    @staticmethod
    def _check_dates(start_date: str, end_date: str) -> None:
        for label, value in (("start_date", start_date),
                             ("end_date", end_date)):
            if not _DATE_RE.match(value or ""):
                raise ConnectorError(
                    f"invalid {label} {value!r}: use YYYY-MM-DD"
                )
        if start_date > end_date:
            raise ConnectorError(
                f"start_date {start_date} is after end_date {end_date}"
            )

    @staticmethod
    def _pretty_type(type_: str, subtype: str) -> str:
        base = (type_ or "").replace("_", " ")
        sub = (subtype or "").replace("_", " ")
        return f"{base} ({sub})".strip() if sub else base

    def _validate_client(self, base: str, cid: str, sec: str) -> None:
        """Cheap credential check: a tiny institutions read."""
        self._api(
            base,
            "/institutions/get",
            {"count": 1, "offset": 0},
            cid=cid,
            secret=sec,
        )

    @staticmethod
    def _token_from_note(note: str) -> str:
        """Pull the Link public_token out of a checkpoint resolution note."""
        text = (note or "").strip()
        if not text:
            return ""
        match = re.search(
            r"public[_-]?token\s*(?:=|:)?\s*[\"']?([A-Za-z0-9_.\-]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        tokens = text.split()
        if len(tokens) == 1:
            return tokens[0].strip("\"'")
        return ""
