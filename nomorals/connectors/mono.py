"""Mono connector — Nigerian (African) open banking.

Mono (https://mono.co, docs: https://docs.mono.co) is the "Plaid for Africa":
with the owner's consent it links their bank accounts through the Mono
Connect widget and exposes account details, transactions, identity/KYC,
income analysis, and bank statements through one REST API.

Auth: one secret key (dashboard: https://app.withmono.com, Settings ->
API Keys; ``test_sk_*`` for sandbox, ``live_sk_*`` for production), sent on
every server-to-server call in the ``mono-sec-key`` header.
``AuthMethod.API_KEY``.

Account linking is a human-in-the-loop flow: the owner completes the Mono
Connect widget in their own browser (entering their own bank credentials /
OTP there — Devon never sees them), the widget hands back a short-lived
authorization ``code``, and Devon exchanges that code for a permanent Mono
account id via ``POST /accounts/auth``. Devon guides the owner; it never
fakes a linking.

Amounts come back in minor units (kobo for NGN, pesewa for GHS, cents for
KES/ZAR); the ``summarize_*`` helpers convert to major units. Account
numbers and BVNs are masked to their last four digits in summaries —
never shown in full.

READ-ONLY for money: this connector reads financial data. Payments go
through Mono DirectPay / Direct Debit, which need a separate product setup
and are deliberately not implemented here.
"""

from __future__ import annotations

import re
import time
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

__all__ = ["MonoConnector", "MonoError"]

_log = get_logger(__name__)

API_BASE = "https://api.withmono.com/v2"
SECRET_ENV = "MONO_SECRET_KEY"
DOCS_URL = "https://docs.mono.co"

#: Currencies whose Mono amounts are reported in minor units.
_MINOR_UNIT_DIVISOR = {
    "NGN": 100,  # kobo
    "GHS": 100,  # pesewa
    "KES": 100,  # cents
    "ZAR": 100,  # cents
}


class MonoError(ConnectorError):
    """A Mono API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        mono_code: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.mono_code = mono_code


@register_connector
class MonoConnector(Connector):
    """Devon's Mono adapter: African open-banking financial data."""

    id = "mono"
    name = "Mono"
    description = (
        "Nigerian/African open banking: link bank accounts through the Mono "
        "Connect widget, then read account details, transactions, identity, "
        "income, and statements. Authenticates with a Mono secret key "
        "(mono-sec-key header). Read-only."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        secret_key: str | None = None,
        code: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Store the Mono secret key; optionally link an account now.

        ``secret_key`` (or the ``MONO_SECRET_KEY`` env var, or a secure
        prompt) is validated against the API and vault-stored. When ``code``
        — the authorization code from a completed Mono Connect widget
        session — is also given, it is exchanged for an account id and that
        account is linked in the same step. Without a code, the owner gets
        the linking guide and can finish with ``link_account(code=...)``;
        with ``db`` a human checkpoint is opened instead so the flow pauses
        cleanly until the owner links in their browser.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "mono is already connected — one account per service. "
                "Disconnect first to switch secret keys."
            )
        secret = (secret_key or "").strip() or prompt_secret(
            "Mono secret key", env_var=SECRET_ENV
        )
        if not secret:
            raise ConnectorError("empty secret key: nothing to connect with")
        self._validate_key(secret)
        linked: list[dict[str, Any]] = []
        account_label = "mono"
        if code:
            account_id = self._exchange_code(secret, code)
            info = self._account_info(secret, account_id)
            linked.append(self._link_record(account_id, info))
            account_label = self._account_label(info)
        self._store_credential(
            "mono",
            secret,
            credential_type="api_key",
            scopes=["accounts", "transactions", "identity", "income",
                    "statements"],
            metadata={"linked_accounts": linked},
        )
        _log.info("mono connected (%d linked account(s))", len(linked))
        if code:
            message = (
                f"connected to Mono and linked {account_label}. The secret "
                "key is in the encrypted vault."
            )
        else:
            message = (
                "connected to Mono (secret key validated and vault-stored). "
                "No bank account linked yet — complete the Mono Connect "
                "widget in your browser, then call "
                "link_account(code=<code from the widget>)."
            )
        return ConnectResult(
            ok=True,
            account=account_label,
            scopes=["accounts", "transactions", "identity", "income",
                    "statements"],
            message=message,
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name mono`",
            )
        try:
            self._api("GET", "/institutions", secret=cred.password)
        except MonoError as exc:
            return ConnectorStatus(
                connected=False,
                account="mono",
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"secret key rejected ({exc}): reconnect with a fresh key",
            )
        linked = (cred.metadata or {}).get("linked_accounts", [])
        names = [a.get("label", a.get("account_id", "?")) for a in linked]
        return ConnectorStatus(
            connected=True,
            account=", ".join(names) if names else "mono (no accounts linked)",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="secret key valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/institutions", secret=cred.password)
            return True
        except ConnectorError:
            return False

    # ── account linking (human-in-the-loop) ──────────────────────

    def begin_link(
        self,
        *,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Start the human bank-linking flow.

        Prints the Mono Connect widget steps for the owner. With ``db`` the
        flow pauses at a human checkpoint (the owner links in their own
        browser, then resolves the checkpoint with the widget's ``code``);
        without it, the guide is printed and the owner hands the code back
        to ``link_account(code=...)``. Devon never sees bank credentials.
        """
        from .checkpoints import CheckpointKind, HumanCheckpointPending

        self._require_credential()
        guide = self.link_instructions()
        print(guide)
        if db is None:
            return {
                "linked": False,
                "next": "complete the widget, then call link_account(code=<code>)",
            }
        try:
            self.request_human(
                CheckpointKind.MANUAL_STEP,
                "Link a bank account with Mono Connect",
                guide
                + "\n\nWhen the widget reports success, resolve this "
                "checkpoint with the authorization code, e.g. note "
                "'code=<code from the widget>'.",
                db=db,
                context=context,
                resume_state={"stage": "link_account"},
            )
        except HumanCheckpointPending as pending:
            # Non-interactive: the checkpoint is persisted; surface the id.
            return {
                "linked": False,
                "checkpoint_id": pending.checkpoint.id,
                "next": (
                    "complete the widget in your browser, then resolve the "
                    f"checkpoint {pending.checkpoint.id} with the code"
                ),
            }
        # Interactive TTY: request_human resolved already — but the code
        # arrives via the checkpoint note, so resume through it.
        raise ConnectorError(
            "linking confirmed interactively but no code was captured — "
            "call link_account(code=<code from the widget>)"
        )

    def link_account(self, code: str) -> dict[str, Any]:
        """Exchange a Connect-widget ``code`` and link the account.

        Returns the linked account's summary. The account id is recorded in
        the vault alongside the secret key so ``linked_accounts()`` knows it.
        """
        cred = self._require_credential()
        account_id = self._exchange_code(cred.password, code)
        info = self._account_info(cred.password, account_id)
        record = self._link_record(account_id, info)
        linked = list((cred.metadata or {}).get("linked_accounts", []))
        if not any(a.get("account_id") == account_id for a in linked):
            linked.append(record)
        self._save_linked(cred, linked)
        _log.info("mono account linked: %s", record["label"])
        return record

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue linking after the owner resolved the human checkpoint."""
        from .checkpoints import CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must finish the Connect widget "
                "first"
            )
        if (checkpoint.resume_state or {}).get("stage") != "link_account":
            raise ConnectorError(
                "mono cannot resume checkpoint stage "
                f"{(checkpoint.resume_state or {}).get('stage')!r}"
            )
        code = self._code_from_note(checkpoint.result_note or "")
        if not code:
            raise ConnectorError(
                "the resolved checkpoint has no authorization code — "
                "resolve it again with note 'code=<code from the widget>'"
            )
        return self.link_account(code)

    # ── reads ────────────────────────────────────────────────────

    def linked_accounts(self) -> list[dict[str, Any]]:
        """Accounts linked through Devon, with live details refreshed."""
        cred = self._require_credential()
        linked = list((cred.metadata or {}).get("linked_accounts", []))
        out = []
        for entry in linked:
            account_id = entry.get("account_id", "")
            try:
                info = self._account_info(cred.password, account_id)
                out.append(self._link_record(account_id, info))
            except MonoError:
                out.append(entry)  # keep the stale record, don't hide it
        return out

    def get_account(self, account_id: str = "") -> dict[str, Any]:
        """Account details (``GET /accounts/{id}``)."""
        cred = self._require_credential()
        return self._account_info(cred.password, self._account_or_default(
            cred, account_id))

    def get_transactions(
        self,
        account_id: str = "",
        *,
        start: str = "",
        end: str = "",
        type: str = "",  # noqa: A002 - matches the Mono API param name
        narration: str = "",
        limit: int = 50,
        page: int = 1,
        paginate: bool = True,
    ) -> dict[str, Any]:
        """Transactions (``GET /accounts/{id}/transactions``).

        ``start``/``end`` are ``YYYY-MM-DD``; ``type`` is ``debit``,
        ``credit``, or empty for both. Amounts are in minor units.
        """
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        params: dict[str, Any] = {
            "paginate": str(paginate).lower(),
            "limit": max(1, min(limit, 100)),
            "page": max(1, page),
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if type in ("debit", "credit"):
            params["type"] = type
        elif type:
            raise ConnectorError(
                f"invalid transaction type {type!r}: use 'debit' or 'credit'"
            )
        if narration:
            params["narration"] = narration
        data = self._api(
            "GET", f"/accounts/{aid}/transactions",
            params=params, secret=cred.password,
        )
        return {
            "account_id": aid,
            "transactions": data.get("data", []),
            "meta": data.get("meta", {}),
        }

    def get_credits(
        self, account_id: str = "", *, limit: int = 50, page: int = 1
    ) -> dict[str, Any]:
        """Historical credits (``GET /accounts/{id}/credits``)."""
        return self._directional(account_id, "credits", limit=limit, page=page)

    def get_debits(
        self, account_id: str = "", *, limit: int = 50, page: int = 1
    ) -> dict[str, Any]:
        """Historical debits (``GET /accounts/{id}/debits``)."""
        return self._directional(account_id, "debits", limit=limit, page=page)

    def get_identity(self, account_id: str = "") -> dict[str, Any]:
        """Identity/KYC overview (``GET /accounts/{id}/identity``).

        Not every institution returns identity data. The raw payload may
        contain a BVN — use :meth:`summarize_identity` for an owner-safe
        view (last four digits only).
        """
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "GET", f"/accounts/{aid}/identity", secret=cred.password
        )
        return {"account_id": aid, "identity": data.get("data", {})}

    def get_income(self, account_id: str = "") -> dict[str, Any]:
        """Income analysis (``GET /accounts/{id}/income``).

        Mono's figure is an estimate with a confidence interval — not a
        certified salary.
        """
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "GET", f"/accounts/{aid}/income", secret=cred.password
        )
        return {"account_id": aid, "income": data.get("data", {})}

    def get_statement(
        self,
        account_id: str = "",
        *,
        output: str = "json",
        period: str = "last3months",
    ) -> dict[str, Any]:
        """Bank statement (``GET /accounts/{id}/statement``).

        ``output`` is ``json`` or ``pdf``; ``period`` like
        ``last3months``/``last6months``/``last12months`` (1–12 months per
        call). A ``pdf`` request starts a generation job — poll it with
        :meth:`poll_statement_pdf`.
        """
        if output not in ("json", "pdf"):
            raise ConnectorError(
                f"invalid statement output {output!r}: use 'json' or 'pdf'"
            )
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "GET",
            f"/accounts/{aid}/statement",
            params={"output": output, "period": period},
            secret=cred.password,
        )
        return {"account_id": aid, "output": output, "statement": data.get("data", {})}

    def poll_statement_pdf(
        self, account_id: str, job_id: str
    ) -> dict[str, Any]:
        """Poll a PDF statement generation job to completion."""
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "GET",
            f"/accounts/{aid}/statement/{job_id}",
            secret=cred.password,
        )
        return {"account_id": aid, "job_id": job_id,
                "status": data.get("data", {})}

    def sync_data(self, account_id: str = "") -> dict[str, Any]:
        """Trigger a manual data refresh (``POST /accounts/{id}/sync``).

        Some institutions ask the owner to re-authorize first; then use
        :meth:`reauthorise` to get a widget token for them.
        """
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "POST", f"/accounts/{aid}/sync", secret=cred.password
        )
        return {"account_id": aid, "result": data.get("data", data)}

    def reauthorise(self, account_id: str = "") -> dict[str, Any]:
        """Get a re-auth token for the Connect widget (human step).

        The token expires in 10 minutes; the owner completes verification
        in the widget themselves.
        """
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "POST", f"/accounts/{aid}/reauthorise", secret=cred.password
        )
        payload = data.get("data", data)
        return {"account_id": aid, "token": payload.get("token", ""),
                "note": "pass this token to the Mono Connect widget; "
                        "the owner completes verification there"}

    def unlink_account(self, account_id: str = "") -> dict[str, Any]:
        """Unlink an account (``POST /accounts/{id}/unlink``) and forget it."""
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "POST", f"/accounts/{aid}/unlink", secret=cred.password
        )
        linked = [
            a for a in (cred.metadata or {}).get("linked_accounts", [])
            if a.get("account_id") != aid
        ]
        self._save_linked(cred, linked)
        _log.info("mono account unlinked: %s", aid)
        return {"account_id": aid, "unlinked": True,
                "result": data.get("data", data)}

    def get_institutions(self) -> list[dict[str, Any]]:
        """Financial institutions available on Mono (``GET /institutions``)."""
        cred = self._require_credential()
        data = self._api("GET", "/institutions", secret=cred.password)
        items = data.get("data", [])
        return items if isinstance(items, list) else []

    # ── owner-safe summaries ─────────────────────────────────────

    def summarize_account(self, account_id: str = "") -> dict[str, Any]:
        """Account overview in major currency units, identifiers masked."""
        info = self.get_account(account_id)
        acct = info.get("account", {})
        currency = str(acct.get("currency", "NGN"))
        return {
            "account_id": info.get("account_id", ""),
            "name": acct.get("name", ""),
            "number": self._mask(acct.get("accountNumber")
                                or acct.get("account_number", "")),
            "type": acct.get("type", ""),
            "currency": currency,
            "balance": self._major(acct.get("balance", 0), currency),
            "institution": (acct.get("institution") or {}).get("name", ""),
            "data_status": (info.get("meta") or {}).get("data_status", ""),
        }

    def summarize_identity(self, account_id: str = "") -> dict[str, Any]:
        """Identity with BVN / account numbers reduced to last four digits."""
        raw = self.get_identity(account_id)
        ident = dict(raw.get("identity", {}))
        for key in ("bvn", "accountNumber", "account_number", "phoneNumber",
                    "phone_number"):
            if ident.get(key):
                ident[key] = self._mask(str(ident[key]))
        return {"account_id": raw.get("account_id", ""), "identity": ident}

    def summarize_transactions(
        self,
        account_id: str = "",
        *,
        start: str = "",
        end: str = "",
        type: str = "",  # noqa: A002 - matches the Mono API param name
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Transactions in major units with plain narration/type/date."""
        result = self.get_transactions(
            account_id, start=start, end=end, type=type, limit=limit
        )
        currency = "NGN"
        out = []
        for txn in result.get("transactions", []):
            currency = str(txn.get("currency", currency))
            out.append({
                "date": txn.get("date", ""),
                "type": txn.get("type", ""),
                "narration": txn.get("narration", ""),
                "amount": self._major(txn.get("amount", 0), currency),
                "currency": currency,
                "category": txn.get("category", ""),
                "balance_after": self._major(txn.get("balance", 0), currency),
            })
        return out

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "mono is not connected — run "
                "`nm connectors connect --name mono` first"
            )
        return cred

    def _headers(self, secret: str) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "mono-sec-key": secret,
        }

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        secret: str | None = None,
    ) -> dict[str, Any]:
        """One Mono API call; errors become MonoError with status."""
        key = secret or self._require_credential().password
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=self._headers(key), params=params
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=self._headers(key)
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise MonoError(f"mono request failed: {exc}") from exc
        if resp.status == 401:
            raise MonoError(
                "mono rejected the secret key (401): it is invalid, "
                "expired, or revoked — reconnect with a fresh key "
                "from https://app.withmono.com",
                status_code=401,
            )
        if resp.status == 429:
            raise MonoError(
                "mono rate limit exceeded (429) — wait a minute and retry",
                status_code=429,
            )
        if not resp.ok:
            try:
                body = resp.json()
                detail = str(body.get("message", body))[:200]
                code = str(body.get("code", ""))
            except Exception:  # noqa: BLE001 - fall back to raw text
                detail, code = resp.text[:200], ""
            raise MonoError(
                f"mono {method} {path} failed ({resp.status}): {detail}",
                status_code=resp.status,
                mono_code=code,
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise MonoError(
                f"mono {method} {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {"data": data}

    # ── helpers ──────────────────────────────────────────────────

    def _validate_key(self, secret: str) -> None:
        """A cheap authenticated call: bad keys fail fast here."""
        self._api("GET", "/institutions", secret=secret)

    def _exchange_code(self, secret: str, code: str) -> str:
        """Swap a Connect-widget code for a permanent account id."""
        code = (code or "").strip()
        if not code:
            raise ConnectorError(
                "empty authorization code — complete the Mono Connect "
                "widget first, then pass its code"
            )
        data = self._api(
            "POST", "/accounts/auth", {"code": code}, secret=secret
        )
        account_id = str(data.get("id", "") or (data.get("data") or {}).get("id", ""))
        if not account_id:
            raise MonoError(
                "mono did not return an account id for this code — "
                "the code may have expired (finish the exchange promptly "
                "after the widget succeeds)"
            )
        return account_id

    def _account_info(self, secret: str, account_id: str) -> dict[str, Any]:
        data = self._api("GET", f"/accounts/{account_id}", secret=secret)
        return {
            "account_id": account_id,
            "account": (data.get("data") or {}).get("account", {}),
            "meta": (data.get("data") or {}).get("meta", {}),
        }

    def _account_or_default(self, cred: Any, account_id: str) -> str:
        aid = (account_id or "").strip()
        if aid:
            return aid
        linked = (cred.metadata or {}).get("linked_accounts", [])
        if len(linked) == 1:
            return str(linked[0].get("account_id", ""))
        if not linked:
            raise ConnectorError(
                "no bank account linked yet — complete the Mono Connect "
                "widget, then call link_account(code=<code>)"
            )
        raise ConnectorError(
            f"{len(linked)} accounts linked — pass account_id explicitly "
            "(see linked_accounts())"
        )

    def _link_record(
        self, account_id: str, info: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "account_id": account_id,
            "label": self._account_label(info),
            "linked_at": time.time(),
        }

    @staticmethod
    def _account_label(info: dict[str, Any]) -> str:
        acct = info.get("account", {})
        name = acct.get("name", "")
        inst = (acct.get("institution") or {}).get("name", "")
        num = MonoConnector._mask(
            str(acct.get("accountNumber") or acct.get("account_number") or "")
        )
        parts = [p for p in (name, f"({inst})" if inst else "", num) if p]
        return " ".join(parts) or info.get("account_id", "")

    def _save_linked(self, cred: Any, linked: list[dict[str, Any]]) -> None:
        meta = dict(cred.metadata or {})
        meta["linked_accounts"] = linked
        self._store_credential(
            cred.username,
            cred.password,
            credential_type="api_key",
            scopes=list(meta.get("scopes", [])),
            metadata=meta,
        )

    @staticmethod
    def _major(amount: Any, currency: str) -> float:
        try:
            value = float(amount or 0)
        except (TypeError, ValueError):
            return 0.0
        return value / _MINOR_UNIT_DIVISOR.get(currency.upper(), 100)

    @staticmethod
    def _mask(value: str) -> str:
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) <= 4:
            return "…" + digits if digits else ""
        return "…" + digits[-4:]

    def _directional(
        self, account_id: str, kind: str, *, limit: int, page: int
    ) -> dict[str, Any]:
        cred = self._require_credential()
        aid = self._account_or_default(cred, account_id)
        data = self._api(
            "GET",
            f"/accounts/{aid}/{kind}",
            params={"limit": max(1, min(limit, 100)), "page": max(1, page)},
            secret=cred.password,
        )
        return {
            "account_id": aid,
            kind: data.get("data", []),
            "meta": data.get("meta", {}),
        }

    @staticmethod
    def _code_from_note(note: str) -> str:
        """Pull the widget code out of a checkpoint resolution note."""
        text = (note or "").strip()
        if not text:
            return ""
        match = re.search(
            r"code\s*(?:=|:|\bis\b)?\s*[\"']?([A-Za-z0-9_.\-]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        tokens = text.split()
        if len(tokens) == 1 and tokens[0].lower() != "code":
            return tokens[0].strip("\"'")
        return ""

    def link_instructions(self) -> str:
        """The human steps for the part only the owner can do: link a bank."""
        return "\n".join([
            "Link a bank account with Mono Connect (only you can do this):",
            "1. You need a Mono app: sign up at https://app.withmono.com and",
            "   copy your PUBLIC key (dashboard -> API Keys).",
            "2. Open the Mono Connect widget with that public key — e.g.:",
            "     <script src=\"https://connect.mono.co/connect.js\"></script>",
            "     new Connect({ key: '<PUBLIC_KEY>', scope: 'auth',",
            "       onSuccess: ({code}) => /* give this code to Devon */ });",
            "   Full guide: https://docs.mono.co/docs/financial-data/overview",
            "3. Pick your bank in the widget and sign in there yourself.",
            "   Your bank credentials/OTP stay in the widget — Devon never",
            "   sees them.",
            "4. The widget calls onSuccess with a short-lived code. Hand that",
            "   code to Devon: link_account(code=<code>). Devon exchanges it",
            "   for a permanent account id (POST /accounts/auth) and links",
            "   the account.",
        ])
