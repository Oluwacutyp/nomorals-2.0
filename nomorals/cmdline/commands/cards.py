"""``nm cards`` — virtual card management."""

from __future__ import annotations

import argparse
import os
from typing import Any
from ..emit import _emit


def _cmd_cards(args: argparse.Namespace, context: Any) -> int:
    """Virtual card management via Flutterwave."""
    from ...connectors.virtualcards import VirtualCardsConnector

    action = getattr(args, "action", "list")
    token = getattr(args, "token", "")
    limit = getattr(args, "limit", 0)

    db = getattr(context, "db", None)
    vault = None
    try:
        from ...accounts.vault import CredentialVault
        vault = CredentialVault(
            db, os.environ.get("NM_VAULT_PASSPHRASE", "")
        )
    except Exception:
        pass

    connector = VirtualCardsConnector(
        vault=vault,  # type: ignore[arg-type]
        db=db,
    )

    if action == "status":
        status = connector.status()
        d = status.to_dict() if hasattr(status, "to_dict") else {"status": str(status)}
        connected = d.get("connected", False)
        _emit(args, d, f"Status: {'connected' if connected else 'disconnected'}")
        return 0

    if action == "list":
        try:
            cards = connector.list_cards()
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed to list cards: {exc}")
            return 1
        _emit(args, {"count": len(cards), "cards": cards},
              f"Found {len(cards)} card(s)")
        return 0

    if action == "create":
        try:
            card = connector.create_card(
                amount=(limit or 10000) / 100.0,
            )
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed to create card: {exc}")
            return 1
        _emit(args, card, f"Created card: {card.get('id', '?')}")
        return 0

    _emit(args, {"error": f"unknown action: {action}"},
          f"Unknown action: {action}. Use: status|list|create")
    return 1
