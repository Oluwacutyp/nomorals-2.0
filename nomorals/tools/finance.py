"""Finance tool — bank account linking, balance, transactions.

Unified facade for Mono (Nigerian banks) and Plaid (US/EU banks).

Actions:
- link(provider) → connect_url
- exchange(provider, code) → account_id
- accounts() → list of accounts
- balance(account_id) → balance info
- transactions(account_id, days) → transaction list
- verify_webhook(provider, payload, signature) → bool
"""

from __future__ import annotations

from typing import Any

from ..connectors.finance import Finance, MonoConnector, PlaidConnector
from ..connectors.vault import CredentialVault
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["register"]

_log = get_logger(__name__)

_finance: Finance | None = None


def _get_finance() -> Finance:
    """Get or create the finance facade."""
    global _finance
    if _finance is None:
        vault = CredentialVault()
        _finance = Finance(vault=vault)
    return _finance


def register(registry: Any) -> None:
    """Register finance tools."""
    
    @registry.register(
        "finance",
        description=(
            "Bank account linking, balance, and transactions via Mono (Nigerian banks) "
            "or Plaid (US/EU banks). Actions: link, exchange, accounts, balance, transactions"
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — link | exchange | accounts | balance | transactions",
            "provider": "str (optional) — mono | plaid | auto",
            "code": "str (optional) — auth code from connect widget",
            "account_id": "str (optional) — account ID for balance/transactions",
            "days": "int (optional, default 30) — transaction history days",
        },
    )
    def finance(action: str, *, provider: str = "auto", code: str = "", 
                account_id: str = "", days: int = 30) -> dict[str, Any]:
        f = _get_finance()
        
        if action == "link":
            return f.link(provider)
        
        elif action == "exchange":
            if not code:
                raise ToolError("exchange requires a code from the connect widget")
            if provider == "auto" or provider == "mono":
                account_id = f.mono.exchange_code(code)
                return {"account_id": account_id, "provider": "mono"}
            else:
                access_token = f.plaid.exchange_public_token(code)
                return {"access_token": access_token, "provider": "plaid"}
        
        elif action == "accounts":
            accounts = f.accounts()
            return {"count": len(accounts), "accounts": accounts}
        
        elif action == "balance":
            if not account_id:
                raise ToolError("balance requires an account_id")
            accounts = f.accounts()
            for acc in accounts:
                if acc["account_id"] == account_id:
                    return acc
            raise ToolError(f"account {account_id} not found")
        
        elif action == "transactions":
            if not account_id:
                raise ToolError("transactions requires an account_id")
            txs = f.transactions(account_id, days)
            return {"count": len(txs), "transactions": txs}
        
        elif action == "status":
            mono_status = f.mono.status()
            plaid_status = f.plaid.status()
            return {
                "mono": mono_status.to_dict(),
                "plaid": plaid_status.to_dict(),
            }
        
        else:
            raise ToolError(f"unknown action: {action}")
