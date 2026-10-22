"""``nm cards`` — virtual card management."""

from __future__ import annotations

import argparse
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
        from ...accounts.vault import open_vault
        vault = open_vault(db, context=context, actor="cards-cli")
    except Exception:
        pass

    connector = VirtualCardsConnector(
        vault=vault,  # type: ignore[arg-type]
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

    if action in ("pause", "close"):
        # pause = the connector's reversible block_card; close = its
        # permanent terminate_card. --token carries the card id (it is
        # the only card identifier the parser offers). Both actions
        # were in the argparse choices but fell through to "unknown
        # action" — the connector methods were never called.
        if not token:
            _emit(args, {"error": "card token required"},
                  f"Usage: nm cards {action} --token <card-id>")
            return 1
        try:
            if action == "pause":
                card = connector.block_card(token)
            else:
                card = connector.terminate_card(token)
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed to {action} card: {exc}")
            return 1
        verb = "Paused" if action == "pause" else "Closed"
        card_id = card.get("id", token) if isinstance(card, dict) else token
        _emit(args, card, f"{verb} card: {card_id}")
        return 0

    _emit(args, {"error": f"unknown action: {action}"},
          f"Unknown action: {action}. Use: status|list|create|pause|close")
    return 1
