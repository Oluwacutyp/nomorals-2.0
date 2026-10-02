"""``nm finance`` — market data and trade ideas."""

from __future__ import annotations

import argparse
from typing import Any
from ..emit import _emit



def _cmd_finance(args: argparse.Namespace, context: Any) -> int:
    """Bank account linking and transactions."""
    from ...connectors.finance import Finance
    from ...connectors.vault import CredentialVault
    
    action = getattr(args, "action", "status")
    provider = getattr(args, "provider", "auto")
    account_id = getattr(args, "account_id", "")
    days = getattr(args, "days", 30)
    
    vault = CredentialVault()
    finance = Finance(vault=vault)
    
    if action == "status":
        mono_status = finance.mono.status()
        plaid_status = finance.plaid.status()
        result = {
            "mono": mono_status.to_dict(),
            "plaid": plaid_status.to_dict(),
        }
        _emit(args, result, f"Mono: {'connected' if mono_status.connected else 'disconnected'}\n"
                           f"Plaid: {'connected' if plaid_status.connected else 'disconnected'}")
    
    elif action == "link":
        result = finance.link(provider)
        _emit(args, result, f"Connect URL: {result.get('connect_url', 'N/A')}")
    
    elif action == "accounts":
        accounts = finance.accounts()
        _emit(args, {"count": len(accounts), "accounts": accounts},
              f"Found {len(accounts)} account(s)")
    
    elif action == "transactions":
        if not account_id:
            _emit(args, {"error": "account_id required"}, "Usage: nm finance transactions --account-id <id>")
            return 1
        txs = finance.transactions(account_id, days)
        _emit(args, {"count": len(txs), "transactions": txs[:10]},
              f"Found {len(txs)} transaction(s)")
    
    elif action == "balance":
        if not account_id:
            _emit(args, {"error": "account_id required"}, "Usage: nm finance balance --account-id <id>")
            return 1
        accounts = finance.accounts()
        for acc in accounts:
            if acc["account_id"] == account_id:
                _emit(args, acc, f"Balance: {acc.get('balance_minor', 0) / 100:.2f} {acc.get('currency', 'NGN')}")
                return 0
        _emit(args, {"error": "account not found"}, f"Account {account_id} not found")
        return 1
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0
