"""Conversational money movement — "send 5k to Mama" as a chat primitive (#49).

Flow:
  parse → resolve recipient → guard check → stage → biometric → execute.

Money moves ONLY after biometric approval (policy.approve_with_biometric),
never from text/voice alone. Unknown recipients are asked about, never
guessed. Warnings are advisory (guard.py) — explicit "send it anyway"
executes per the #43 override rule. Every step is ledger-logged.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..voice.money import TransferStaging, parse_amount as _voice_parse_amount
from .budgets import finance_paths
from .guard import Warning, check_outgoing
from .ledger import Ledger, format_naira, parse_amount

_log = logging.getLogger(__name__)

__all__ = [
    "RecipientStore",
    "parse_send_request",
    "resolve_recipient",
    "send_money",
    "confirm_send",
    "confirm_send_otp",
]

#: Capability required for money movement (biometric-gated per policy).
MONEY_CAPABILITY = "money_transfer"


class RecipientStore:
    """Named transfer recipients with bank details.

    JSON at ~/.nomorals/finance/recipients.json:
      {"mama": {"account_number": "0123456789", "bank_code": "058",
                "bank_name": "GTBank", "added_at": ...}}
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            _, budgets_path = finance_paths(None)
            path = Path(budgets_path).parent / "recipients.json"
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._load(), indent=2), encoding="utf-8")

    def get(self, name: str) -> dict[str, Any] | None:
        """Case-insensitive lookup. Returns the recipient dict or None."""
        key = (name or "").strip().lower()
        return self._load().get(key)

    def add(self, name: str, *, account_number: str, bank_code: str,
            bank_name: str = "") -> dict[str, Any]:
        """Save a recipient. Never raises on bad input — validates first."""
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("recipient name required")
        if not (account_number or "").strip().isdigit():
            raise ValueError("account_number must be digits")
        if not (bank_code or "").strip():
            raise ValueError("bank_code required")
        rec = {
            "name": (name or "").strip(),
            "account_number": account_number.strip(),
            "bank_code": bank_code.strip(),
            "bank_name": bank_name.strip(),
            "added_at": time.time(),
        }
        self._load()[key] = rec
        self._save()
        return rec

    def names(self) -> list[str]:
        return [r["name"] for r in self._load().values()]


def parse_send_request(text: str) -> dict[str, Any] | None:
    """Parse "send 5k to Mama" → {amount_kobo, to} or None.

    Shares parsing with the #16 voice money path.
    """
    from ..voice.money import parse_voice_money
    intent = parse_voice_money(text or "")
    if intent.kind != "transfer" or not intent.amount_kobo:
        return None
    return {
        "amount_kobo": intent.amount_kobo,
        "to": (intent.recipient or "").strip(),
    }


def resolve_recipient(
    name: str,
    store: RecipientStore | None = None,
) -> dict[str, Any]:
    """Resolve a recipient name → {"ok", "recipient"} or {"ok": False, "ask"}.

    Unknown recipients are asked about — never guessed.
    """
    store = store or RecipientStore()
    rec = store.get(name)
    if rec is not None:
        return {"ok": True, "recipient": rec}
    known = store.names()
    hint = f" I know: {', '.join(known)}." if known else ""
    return {
        "ok": False,
        "needs": "recipient",
        "ask": (
            f"I don't have bank details for {name.strip() or 'them'} yet. "
            f"What's their account number and bank?{hint}"
        ),
    }


def _staging(settings: Any = None) -> TransferStaging:
    _, budgets_path = finance_paths(settings)
    path = Path(budgets_path).parent / "staged_transfers.jsonl"
    return TransferStaging(path=path)


def send_money(
    to: str,
    amount: Any,
    *,
    note: str = "",
    context: Any = None,
    ledger: Ledger | None = None,
    recipients: RecipientStore | None = None,
    override_warning: bool = False,
) -> dict[str, Any]:
    """Stage an outgoing transfer. Returns a result dict (never raises).

    Possible outcomes:
      {"ok": False, "needs": "recipient", "ask": ...}  — unknown recipient
      {"ok": False, "warning": {...}}                   — guard flagged it;
        show with [Send anyway] [Cancel]; retry with override_warning=True
      {"ok": False, "needs": "biometric", "staged_id": ...,
       "prompt": ...}                                   — staged; caller must
        run the biometric prompt, then call confirm_send()
      {"ok": False, "error": ...}                       — failed cleanly
    """
    settings = getattr(context, "settings", None)
    ledger = ledger or Ledger(finance_paths(settings)[0])
    recipients = recipients or RecipientStore()

    # 1. amount
    kobo = parse_amount(str(amount)) if not isinstance(amount, int) else int(amount) * 100
    if kobo is None or kobo <= 0:
        return {"ok": False, "error": f"couldn't parse amount {amount!r}"}

    # 2. recipient — asked about, never guessed
    resolved = resolve_recipient(to, recipients)
    if not resolved["ok"]:
        return resolved
    recipient = resolved["recipient"]

    # 3. guard — advisory, not blocking
    warning: Warning | None = None
    if not override_warning:
        warning = check_outgoing(recipient["name"], kobo, ledger)
    if warning is not None:
        _audit(ledger, kobo, recipient["name"], "warning_shown",
               note=warning.message)
        return {"ok": False, "warning": warning.to_dict(),
                "staged": False}

    # 4. stage (money moves ONLY after biometric — see confirm_send)
    staged = _staging(settings).stage(kobo, recipient["name"], note=note)
    _audit(ledger, kobo, recipient["name"], "staged", note=f"id={staged.id}")
    return {
        "ok": False,
        "needs": "biometric",
        "staged_id": staged.id,
        "amount": format_naira(kobo),
        "to": recipient["name"],
        "prompt": (
            f"Confirm with your fingerprint to send {format_naira(kobo)} "
            f"to {recipient['name']}."
        ),
    }


def confirm_send(
    staged_id: str,
    biometric_token: str | None,
    *,
    context: Any = None,
    ledger: Ledger | None = None,
    paystack: Any = None,
    recipients: RecipientStore | None = None,
) -> dict[str, Any]:
    """Execute a staged transfer after biometric approval.

    Requires the capability-bound token from policy.approve_with_biometric().
    No token → no movement, no exceptions-as-control-flow: a clear error.
    """
    settings = getattr(context, "settings", None)
    ledger = ledger or Ledger(finance_paths(settings)[0])
    if not biometric_token:
        return {"ok": False,
                "error": "biometric approval required — money did not move"}

    staging = _staging(settings)
    rec = next((r for r in staging.pending() if r.id == staged_id), None)
    if rec is None:
        return {"ok": False, "error": f"unknown staged transfer {staged_id!r}"}
    if rec.status == "executed":
        return {"ok": False, "error": "transfer already executed"}
    staging.mark(staged_id, "biometric_confirmed", biometric_token=biometric_token)
    rec.status = "biometric_confirmed"
    rec.biometric_token = biometric_token

    # Execute through the connector. Paystack Transfer API is the real path;
    # anything else fails closed — never fake success.
    result = _execute_via_paystack(rec, paystack, recipients)
    if result["ok"]:
        staging.mark(staged_id, "executed")
        ledger.log(rec.amount_kobo, category="transfer",
                   note=f"transfer to {rec.recipient}",
                   kind="spend", source="paystack")
        _audit(ledger, rec.amount_kobo, rec.recipient, "executed",
               note=f"ref={result.get('reference', '')}")
    else:
        _audit(ledger, rec.amount_kobo, rec.recipient, "execute_failed",
               note=result.get("error", ""))
    return result


def confirm_send_otp(
    staged_id: str,
    transfer_code: str,
    otp: str,
    *,
    context: Any = None,
    ledger: Ledger | None = None,
    paystack: Any = None,
) -> dict[str, Any]:
    """Complete an OTP-gated transfer after ``confirm_send`` reported ``needs: otp``.

    The money was already approved (biometric); this just releases it.
    Returns {"ok": True, ...} or {"ok": False, "error": ...}.
    """
    settings = getattr(context, "settings", None)
    ledger = ledger or Ledger(finance_paths(settings)[0])
    if paystack is None:
        return {"ok": False,
                "error": "no transfer connector configured — money did not move"}
    staging = _staging(settings)
    rec = next((r for r in staging.pending() if r.id == staged_id), None)
    if rec is None:
        return {"ok": False, "error": f"unknown staged transfer {staged_id!r}"}
    try:
        data = paystack.finalize_transfer(transfer_code, otp)
    except Exception as exc:  # noqa: BLE001 - connector errors are results
        _audit(ledger, rec.amount_kobo, rec.recipient, "otp_failed",
               note=str(exc)[:120])
        return {"ok": False, "error": f"OTP finalization failed: {exc}"}
    staging.mark(staged_id, "executed")
    ledger.log(rec.amount_kobo, category="transfer",
               note=f"transfer to {rec.recipient}",
               kind="spend", source="paystack")
    _audit(ledger, rec.amount_kobo, rec.recipient, "executed",
           note=f"ref={data.get('reference', '')} (otp)")
    return {"ok": True, "reference": str(data.get("reference", "")),
            "amount": format_naira(rec.amount_kobo), "to": rec.recipient}


def _execute_via_paystack(rec: Any, paystack: Any,
                          recipients: RecipientStore | None = None) -> dict[str, Any]:
    """Real Paystack transfer through the connector. Fail-closed.

    Uses ``PaystackConnector.create_transfer_recipient`` /
    ``initiate_transfer`` — the connector owns the API surface; this
    module owns the conversational flow (parse → guard → biometric).
    The biometric token on the staged record IS the explicit approval,
    so ``confirmed=True``.
    """
    if paystack is None:
        return {"ok": False, "error":
                "no transfer connector configured — money did not move. "
                "Connect Paystack (transfer API) to enable sends."}
    store = recipients or RecipientStore()
    details = store.get(rec.recipient)
    if details is None:
        return {"ok": False, "error":
                f"no bank details for {rec.recipient!r} — money did not move"}
    try:
        created = paystack.create_transfer_recipient(
            details["account_number"],
            details["bank_code"],
            name=details.get("name") or rec.recipient,
        )
        code = (created or {}).get("recipient_code")
        if not code:
            return {"ok": False, "error":
                    f"Paystack recipient creation failed: {str(created)[:200]}"}
        sent = paystack.initiate_transfer(
            rec.amount_kobo,
            code,
            reason=rec.note or f"Devon transfer to {details['name']}",
            confirmed=True,
            biometric_token=getattr(rec, "biometric_token", None),
        )
        data = sent or {}
        status = str(data.get("status") or "")
        if status == "otp":
            return {"ok": False, "needs": "otp",
                    "transfer_code": str(data.get("transfer_code", "")),
                    "error": "Paystack needs an OTP to release this transfer — "
                             "ask the owner for it, then call confirm_send_otp"}
        return {"ok": True, "reference": str(data.get("reference", "")),
                "amount": format_naira(rec.amount_kobo),
                "to": details["name"]}
    except Exception as exc:  # noqa: BLE001 - connector errors are results
        _log.warning("paystack transfer failed: %s", exc)
        return {"ok": False, "error": f"transfer failed: {exc}"}


def _audit(ledger: Ledger, amount_kobo: int, to: str, event: str,
           note: str = "") -> None:
    """Ledger audit trail for guard/send decisions. Never raises."""
    try:
        ledger.log(1, category="transfer_audit",
                   note=f"[{event}] {format_naira(amount_kobo)} to {to} {note}".strip(),
                   kind="spend", source="guard")
    except Exception:  # noqa: BLE001 - audit must not break the flow
        _log.debug("transfer audit log failed", exc_info=True)
