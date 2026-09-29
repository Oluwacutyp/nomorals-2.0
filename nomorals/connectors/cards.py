"""Virtual cards connector — Privacy.com (US-only).

Create virtual cards for online purchases with spending limits, merchant locks,
and single-use cards. Requires US bank account.

API: https://api.privacy.com/v1
Auth: Authorization: api-key <KEY>

Card types:
- SINGLE_USE: Auto-closes after first charge
- MERCHANT_LOCKED: Locked to first merchant
- UNLOCKED: Works anywhere (default)
- DIGITAL_WALLET: For Apple Pay/Google Pay

Spend limits:
- TRANSACTION: Per-transaction limit
- MONTHLY: Monthly spending limit
- ANNUALLY: Annual spending limit
- FOREVER: Lifetime limit

Security:
- PAN/CVV masked in storage (****1234)
- Full details revealed only at checkout handoff
- No self-generated card numbers (provider APIs only)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .base import BaseConnector, ConnectorStatus

__all__ = ["PrivacyCardsConnector"]

_log = get_logger(__name__)


@dataclass
class VirtualCard:
    """Virtual card details."""
    
    token: str  # Privacy.com card token
    provider: str = "privacy_com"
    card_type: str = "UNLOCKED"  # SINGLE_USE, MERCHANT_LOCKED, UNLOCKED, DIGITAL_WALLET
    state: str = "OPEN"  # OPEN, PAUSED, CLOSED
    last_four: str = ""
    masked_pan: str = ""  # ****1234
    spend_limit: int = 0  # In cents
    spend_limit_duration: str = ""  # TRANSACTION, MONTHLY, ANNUALLY, FOREVER
    memo: str = ""
    created_at: float = 0.0
    
    # Full details (masked in storage, revealed only at checkout)
    pan: str = ""
    cvv: str = ""
    exp_month: str = ""
    exp_year: str = ""
    
    def to_dict(self, reveal: bool = False) -> dict[str, Any]:
        """Convert to dict. If reveal=False, masks PAN/CVV."""
        data = {
            "token": self.token,
            "provider": self.provider,
            "card_type": self.card_type,
            "state": self.state,
            "last_four": self.last_four,
            "masked_pan": self.masked_pan,
            "spend_limit": self.spend_limit,
            "spend_limit_duration": self.spend_limit_duration,
            "memo": self.memo,
            "created_at": self.created_at,
        }
        
        if reveal:
            data.update({
                "pan": self.pan,
                "cvv": self.cvv,
                "exp_month": self.exp_month,
                "exp_year": self.exp_year,
            })
        else:
            # Mask sensitive fields
            data.update({
                "pan": self.masked_pan,
                "cvv": "***",
                "exp_month": "**",
                "exp_year": "**",
            })
        
        return data


class PrivacyCardsConnector(BaseConnector):
    """Privacy.com virtual cards connector (US-only).
    
    API: https://api.privacy.com/v1
    Auth: Authorization: api-key <KEY>
    
    Endpoints:
    - POST /v1/card — Create card
    - GET /v1/card — List cards
    - GET /v1/card/{token} — Get card details
    - PUT /v1/card/{token} — Update card (pause/close)
    - POST /v1/card/{token}/spend-limit — Set spend limit
    
    Card creation:
    - type: SINGLE_USE, MERCHANT_LOCKED, UNLOCKED, DIGITAL_WALLET
    - spend_limit: Amount in cents
    - spend_limit_duration: TRANSACTION, MONTHLY, ANNUALLY, FOREVER
    - memo: Description (optional)
    
    Behaviors:
    - create_for_purchase(merchant, amount) → SINGLE_USE, auto-closes
    - create_for_subscription(merchant, monthly) → MERCHANT_LOCKED
    """
    
    name = "privacy_cards"
    description = "Virtual cards for online purchases (US-only, requires US bank)"
    
    BASE_URL = "https://api.privacy.com/v1"
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(vault, config)
        self.http = HttpClient(timeout=30.0)
    
    def status(self) -> ConnectorStatus:
        """Check if Privacy.com is configured."""
        try:
            creds = self._get_credential("api_key")
            if not creds or not creds.get("key"):
                return ConnectorStatus(
                    connected=False,
                    error="No Privacy.com API key configured"
                )
            
            return ConnectorStatus(
                connected=True,
                account="Privacy.com",
                scopes=["create_card", "list_cards", "manage_cards"]
            )
        
        except Exception as e:
            return ConnectorStatus(connected=False, error=str(e))
    
    def connect_url(self) -> str:
        """Return API key setup URL."""
        return "https://privacy.com/settings/api"
    
    def disconnect(self) -> dict[str, Any]:
        """Clear API key."""
        try:
            self._delete_credential("api_key")
            return {"disconnected": True}
        except Exception as e:
            return {"disconnected": False, "error": str(e)}
    
    def capabilities(self) -> list[str]:
        """List implemented capabilities."""
        return [
            "create_card",
            "list_cards",
            "get_card",
            "pause_card",
            "close_card",
            "set_spend_limit",
        ]
    
    def create_card(
        self,
        card_type: str = "UNLOCKED",
        spend_limit: int = 0,
        spend_limit_duration: str = "",
        memo: str = ""
    ) -> VirtualCard:
        """Create a new virtual card.
        
        Args:
            card_type: SINGLE_USE, MERCHANT_LOCKED, UNLOCKED, DIGITAL_WALLET
            spend_limit: Amount in cents (0 = no limit)
            spend_limit_duration: TRANSACTION, MONTHLY, ANNUALLY, FOREVER
            memo: Description
        
        Returns:
            Created card
        
        Raises:
            Exception: If creation fails
        """
        creds = self._get_credential("api_key")
        if not creds or not creds.get("key"):
            raise Exception("No Privacy.com API key configured")
        
        payload = {"type": card_type}
        if spend_limit > 0 and spend_limit_duration:
            payload["spend_limit"] = spend_limit
            payload["spend_limit_duration"] = spend_limit_duration
        if memo:
            payload["memo"] = memo
        
        response = self.http.post_json(
            f"{self.BASE_URL}/card",
            payload,
            headers={
                "Authorization": f"api-key {creds['key']}",
                "Content-Type": "application/json"
            }
        )
        
        if not response.ok:
            raise Exception(f"Card creation failed: {response.status}")
        
        data = response.json()
        card_data = data.get("card", {})
        
        # Extract card details
        pan = card_data.get("pan", "")
        masked_pan = f"****{pan[-4:]}" if len(pan) >= 4 else "****"
        
        card = VirtualCard(
            token=card_data.get("token", ""),
            provider="privacy_com",
            card_type=card_data.get("type", card_type),
            state=card_data.get("state", "OPEN"),
            last_four=pan[-4:] if len(pan) >= 4 else "",
            masked_pan=masked_pan,
            spend_limit=card_data.get("spend_limit", 0),
            spend_limit_duration=card_data.get("spend_limit_duration", ""),
            memo=card_data.get("memo", memo),
            created_at=time.time(),
            pan=pan,
            cvv=card_data.get("cvv", ""),
            exp_month=str(card_data.get("exp_month", "")),
            exp_year=str(card_data.get("exp_year", ""))
        )
        
        # Store card (masked)
        self._store_card(card)
        
        _log.info("Created card: %s (type=%s, limit=%d cents)", 
                  card.token, card.card_type, card.spend_limit)
        
        return card
    
    def create_for_purchase(self, merchant: str, amount_cents: int) -> VirtualCard:
        """Create single-use card for one-time purchase.
        
        Args:
            merchant: Merchant name (for memo)
            amount_cents: Purchase amount in cents
        
        Returns:
            SINGLE_USE card with exact amount limit
        """
        return self.create_card(
            card_type="SINGLE_USE",
            spend_limit=amount_cents,
            spend_limit_duration="TRANSACTION",
            memo=f"Purchase: {merchant}"
        )
    
    def create_for_subscription(self, merchant: str, monthly_cents: int) -> VirtualCard:
        """Create merchant-locked card for subscription.
        
        Args:
            merchant: Merchant name
            monthly_cents: Monthly subscription amount in cents
        
        Returns:
            MERCHANT_LOCKED card with monthly limit
        """
        return self.create_card(
            card_type="MERCHANT_LOCKED",
            spend_limit=monthly_cents,
            spend_limit_duration="MONTHLY",
            memo=f"Subscription: {merchant}"
        )
    
    def list_cards(self, limit: int = 50) -> list[VirtualCard]:
        """List all cards."""
        creds = self._get_credential("api_key")
        if not creds or not creds.get("key"):
            raise Exception("No Privacy.com API key configured")
        
        response = self.http.get(
            f"{self.BASE_URL}/card?page=1&page_size={limit}",
            headers={"Authorization": f"api-key {creds['key']}"}
        )
        
        if not response.ok:
            raise Exception(f"Failed to list cards: {response.status}")
        
        data = response.json()
        cards_data = data.get("data", [])
        
        cards = []
        for card_data in cards_data:
            pan = card_data.get("pan", "")
            cards.append(VirtualCard(
                token=card_data.get("token", ""),
                provider="privacy_com",
                card_type=card_data.get("type", ""),
                state=card_data.get("state", ""),
                last_four=pan[-4:] if len(pan) >= 4 else "",
                masked_pan=f"****{pan[-4:]}" if len(pan) >= 4 else "****",
                spend_limit=card_data.get("spend_limit", 0),
                spend_limit_duration=card_data.get("spend_limit_duration", ""),
                memo=card_data.get("memo", ""),
                created_at=time.time()
            ))
        
        return cards
    
    def get_card(self, token: str, reveal: bool = False) -> VirtualCard:
        """Get card details.
        
        Args:
            token: Card token
            reveal: If True, returns full PAN/CVV (use only at checkout)
        
        Returns:
            Card details
        """
        creds = self._get_credential("api_key")
        if not creds or not creds.get("key"):
            raise Exception("No Privacy.com API key configured")
        
        response = self.http.get(
            f"{self.BASE_URL}/card/{token}",
            headers={"Authorization": f"api-key {creds['key']}"}
        )
        
        if not response.ok:
            raise Exception(f"Failed to get card: {response.status}")
        
        data = response.json()
        card_data = data.get("card", {})
        
        pan = card_data.get("pan", "")
        return VirtualCard(
            token=card_data.get("token", token),
            provider="privacy_com",
            card_type=card_data.get("type", ""),
            state=card_data.get("state", ""),
            last_four=pan[-4:] if len(pan) >= 4 else "",
            masked_pan=f"****{pan[-4:]}" if len(pan) >= 4 else "****",
            spend_limit=card_data.get("spend_limit", 0),
            spend_limit_duration=card_data.get("spend_limit_duration", ""),
            memo=card_data.get("memo", ""),
            created_at=time.time(),
            pan=pan if reveal else "",
            cvv=card_data.get("cvv", "") if reveal else "",
            exp_month=str(card_data.get("exp_month", "")),
            exp_year=str(card_data.get("exp_year", ""))
        )
    
    def pause_card(self, token: str) -> bool:
        """Pause a card (can be resumed)."""
        return self._update_card_state(token, "PAUSED")
    
    def close_card(self, token: str) -> bool:
        """Close a card permanently."""
        return self._update_card_state(token, "CLOSED")
    
    def set_spend_limit(self, token: str, limit_cents: int, duration: str) -> bool:
        """Set spend limit on a card.
        
        Args:
            token: Card token
            limit_cents: Amount in cents
            duration: TRANSACTION, MONTHLY, ANNUALLY, FOREVER
        
        Returns:
            True if successful
        """
        creds = self._get_credential("api_key")
        if not creds or not creds.get("key"):
            raise Exception("No Privacy.com API key configured")
        
        response = self.http.post_json(
            f"{self.BASE_URL}/card/{token}/spend-limit",
            {
                "spend_limit": limit_cents,
                "spend_limit_duration": duration
            },
            headers={
                "Authorization": f"api-key {creds['key']}",
                "Content-Type": "application/json"
            }
        )
        
        return response.ok
    
    def _update_card_state(self, token: str, state: str) -> bool:
        """Update card state (PAUSED/CLOSED)."""
        creds = self._get_credential("api_key")
        if not creds or not creds.get("key"):
            raise Exception("No Privacy.com API key configured")
        
        response = self.http.put_json(
            f"{self.BASE_URL}/card/{token}",
            {"state": state},
            headers={
                "Authorization": f"api-key {creds['key']}",
                "Content-Type": "application/json"
            }
        )
        
        return response.ok
    
    def _store_card(self, card: VirtualCard) -> None:
        """Store card in vault (masked)."""
        # Store masked version
        masked_data = card.to_dict(reveal=False)
        self._store_credential(f"card_{card.token}", masked_data)
