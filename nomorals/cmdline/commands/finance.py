"""``nm finance`` — bank account linking and transactions (Mono + Plaid)."""

from __future__ import annotations

import argparse
import os
from typing import Any
from ..emit import _emit


def _get_vault(context: Any):
    try:
        from ...accounts.vault import CredentialVault
        return CredentialVault(
            getattr(context, "db", None),
            os.environ.get("NM_VAULT_PASSPHRASE", ""),
        )
    except Exception:
        return None


def _cmd_finance(args: argparse.Namespace, context: Any) -> int:
    """Bank account linking and transactions via Mono and Plaid connectors."""
    from ...connectors.mono import MonoConnector
    from ...connectors.plaid import PlaidConnector

    action = getattr(args, "action", "status")
    provider = getattr(args, "provider", "auto").lower()
    account_id = getattr(args, "account_id", "")

    db = getattr(context, "db", None)
    vault = _get_vault(context)

    connectors: dict[str, Any] = {}
    if provider in ("auto", "mono"):
        try:
            connectors["mono"] = MonoConnector(vault=vault, db=db)  # type: ignore[arg-type]
        except Exception as exc:
            _emit(args, {"error": f"mono init failed: {exc}"}, "")
            return 1
    if provider in ("auto", "plaid"):
        try:
            connectors["plaid"] = PlaidConnector(vault=vault, db=db)  # type: ignore[arg-type]
        except Exception as exc:
            _emit(args, {"error": f"plaid init failed: {exc}"}, "")
            return 1

    if action == "status":
        result = {}
        lines = []
        for name, conn in connectors.items():
            try:
                st = conn.status()
                d = st.to_dict() if hasattr(st, "to_dict") else {"status": str(st)}
            except Exception as exc:
                d = {"connected": False, "error": str(exc)}
            result[name] = d
            lines.append(f"{name.capitalize()}: {'connected' if d.get('connected') else 'disconnected'}")
        _emit(args, result, "\n".join(lines))
        return 0

    # For non-status actions, pick the single connector
    if len(connectors) == 1:
        conn = next(iter(connectors.values()))
        conn_name = next(iter(connectors.keys()))
    elif provider in connectors:
        conn = connectors[provider]
        conn_name = provider
    else:
        _emit(args, {"error": "specify --provider mono|plaid for this action"},
              "Use --provider mono or --provider plaid")
        return 1

    if action == "accounts":
        try:
            accounts = conn.list_accounts() if hasattr(conn, "list_accounts") else conn.accounts()
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed: {exc}")
            return 1
        _emit(args, {"provider": conn_name, "count": len(accounts), "accounts": accounts},
              f"Found {len(accounts)} account(s) via {conn_name}")
        return 0

    if action == "transactions":
        if not account_id:
            _emit(args, {"error": "account_id required"},
                  "Usage: nm finance transactions --provider mono --account-id <id>")
            return 1
        try:
            txs = conn.get_transactions(account_id) if hasattr(conn, "get_transactions") else []
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed: {exc}")
            return 1
        _emit(args, {"count": len(txs), "transactions": txs[:10]},
              f"Found {len(txs)} transaction(s)")
        return 0

    _emit(args, {"error": f"unknown action: {action}"},
          f"Unknown action: {action}. Use: status|accounts|transactions")
    return 1
