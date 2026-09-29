"""Virtual cards tool — create and manage virtual cards via Privacy.com.

Actions:
- create(type, limit, duration, memo) → card
- create_purchase(merchant, amount_cents) → single-use card
- create_subscription(merchant, monthly_cents) → merchant-locked card
- list() → all cards
- get(token, reveal) → card details
- pause(token) → pause card
- close(token) → close card permanently
- set_limit(token, limit_cents, duration) → update limit
"""

from __future__ import annotations

from typing import Any

from ..connectors.cards import PrivacyCardsConnector
from ..connectors.vault import CredentialVault
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["register"]

_log = get_logger(__name__)

_connector: PrivacyCardsConnector | None = None


def _get_connector() -> PrivacyCardsConnector:
    """Get or create the cards connector."""
    global _connector
    if _connector is None:
        vault = CredentialVault()
        _connector = PrivacyCardsConnector(vault=vault)
    return _connector


def register(registry: Any) -> None:
    """Register virtual cards tools."""
    
    @registry.register(
        "cards",
        description=(
            "Virtual cards via Privacy.com (US-only). Create single-use, merchant-locked, "
            "or unlocked cards with spending limits. Actions: create, create_purchase, "
            "create_subscription, list, get, pause, close, set_limit, status"
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — create | create_purchase | create_subscription | list | get | pause | close | set_limit | status",
            "card_type": "str (optional) — SINGLE_USE | MERCHANT_LOCKED | UNLOCKED | DIGITAL_WALLET",
            "merchant": "str (optional) — merchant name for purchase/subscription",
            "amount_cents": "int (optional) — amount in cents for purchase card",
            "monthly_cents": "int (optional) — monthly amount in cents for subscription card",
            "spend_limit": "int (optional) — spend limit in cents",
            "spend_limit_duration": "str (optional) — TRANSACTION | MONTHLY | ANNUALLY | FOREVER",
            "memo": "str (optional) — card description",
            "token": "str (optional) — card token for get/pause/close/set_limit",
            "reveal": "bool (optional, default False) — reveal full PAN/CVV (checkout only)",
            "limit": "int (optional, default 50) — max cards to list",
        },
    )
    def cards(action: str, *, card_type: str = "UNLOCKED", merchant: str = "",
              amount_cents: int = 0, monthly_cents: int = 0,
              spend_limit: int = 0, spend_limit_duration: str = "",
              memo: str = "", token: str = "", reveal: bool = False,
              limit: int = 50) -> dict[str, Any]:
        c = _get_connector()
        
        if action == "status":
            return c.status().to_dict()
        
        elif action == "capabilities":
            return {"capabilities": c.capabilities()}
        
        elif action == "create":
            card = c.create_card(
                card_type=card_type,
                spend_limit=spend_limit,
                spend_limit_duration=spend_limit_duration,
                memo=memo
            )
            return card.to_dict(reveal=reveal)
        
        elif action == "create_purchase":
            if not merchant or amount_cents <= 0:
                raise ToolError("create_purchase requires merchant and amount_cents > 0")
            card = c.create_for_purchase(merchant, amount_cents)
            return card.to_dict(reveal=reveal)
        
        elif action == "create_subscription":
            if not merchant or monthly_cents <= 0:
                raise ToolError("create_subscription requires merchant and monthly_cents > 0")
            card = c.create_for_subscription(merchant, monthly_cents)
            return card.to_dict(reveal=reveal)
        
        elif action == "list":
            card_list = c.list_cards(limit=limit)
            return {
                "count": len(card_list),
                "cards": [card.to_dict() for card in card_list]
            }
        
        elif action == "get":
            if not token:
                raise ToolError("get requires a card token")
            card = c.get_card(token, reveal=reveal)
            return card.to_dict(reveal=reveal)
        
        elif action == "pause":
            if not token:
                raise ToolError("pause requires a card token")
            success = c.pause_card(token)
            return {"paused": success, "token": token}
        
        elif action == "close":
            if not token:
                raise ToolError("close requires a card token")
            success = c.close_card(token)
            return {"closed": success, "token": token}
        
        elif action == "set_limit":
            if not token:
                raise ToolError("set_limit requires a card token")
            if spend_limit <= 0 or not spend_limit_duration:
                raise ToolError("set_limit requires spend_limit > 0 and spend_limit_duration")
            success = c.set_spend_limit(token, spend_limit, spend_limit_duration)
            return {"updated": success, "token": token, "limit": spend_limit, "duration": spend_limit_duration}
        
        elif action == "connect_url":
            return {"connect_url": c.connect_url()}
        
        else:
            raise ToolError(f"unknown action: {action}")
