"""``nm cards`` — card surfaces."""

from __future__ import annotations

import argparse
from typing import Any
from ..emit import _emit



def _cmd_cards(args: argparse.Namespace, context: Any) -> int:
    """Virtual card management."""
    from ...connectors.cards import PrivacyCardsConnector
    from ...connectors.vault import CredentialVault
    
    action = getattr(args, "action", "list")
    token = getattr(args, "token", "")
    card_type = getattr(args, "type", "UNLOCKED")
    limit = getattr(args, "limit", 0)
    merchant = getattr(args, "merchant", "")
    
    vault = CredentialVault()
    connector = PrivacyCardsConnector(vault=vault)
    
    if action == "status":
        status = connector.status()
        _emit(args, status.to_dict(), f"Status: {'connected' if status.connected else 'disconnected'}")
    
    elif action == "list":
        cards = connector.list_cards()
        _emit(args, {"count": len(cards), "cards": [c.to_dict() for c in cards]},
              f"Found {len(cards)} card(s)")
    
    elif action == "create":
        if merchant:
            card = connector.create_for_purchase(merchant, limit or 10000)
        else:
            card = connector.create_card(card_type=card_type, spend_limit=limit)
        _emit(args, card.to_dict(), f"Created card: {card.token}")
    
    elif action == "pause":
        if not token:
            _emit(args, {"error": "token required"}, "Usage: nm cards pause --token <token>")
            return 1
        success = connector.pause_card(token)
        _emit(args, {"paused": success, "token": token}, f"Paused: {success}")
    
    elif action == "close":
        if not token:
            _emit(args, {"error": "token required"}, "Usage: nm cards close --token <token>")
            return 1
        success = connector.close_card(token)
        _emit(args, {"closed": success, "token": token}, f"Closed: {success}")
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0
