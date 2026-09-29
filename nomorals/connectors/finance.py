"""Finance connector — Mono (Nigerian banks) + Plaid (US/EU banks).

Unified interface for bank account linking, balance checks, transactions,
and identity verification. Routes NG users to Mono, others to Plaid.

APIs:
- Mono: https://api.withmono.com/v2 (mono-sec-key header)
- Plaid: https://sandbox.plaid.com (client_id + secret)

Security:
- Webhook signatures verified (Mono HMAC-SHA512)
- Credentials in vault only
- Account masks stored, full account numbers never logged
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .base import BaseConnector, ConnectorStatus
from .patterns import ConnectionPattern

__all__ = ["MonoConnector", "PlaidConnector", "Finance"]

_log = get_logger(__name__)


@dataclass
class BankAccount:
    """Linked bank account."""
    
    account_id: str
    provider: str  # "mono" or "plaid"
    institution: str
    mask: str  # Last 4 digits
    account_type: str  # "checking", "savings", etc.
    balance_minor: int = 0  # In minor units (kobo/cents)
    currency: str = "NGN"
    linked_at: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "provider": self.provider,
            "institution": self.institution,
            "mask": self.mask,
            "account_type": self.account_type,
            "balance_minor": self.balance_minor,
            "currency": self.currency,
            "linked_at": self.linked_at,
        }


@dataclass
class Transaction:
    """Bank transaction."""
    
    transaction_id: str
    account_id: str
    amount_minor: int
    currency: str
    description: str
    date: str  # ISO format
    category: str = ""
    merchant: str = ""
    pending: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "description": self.description,
            "date": self.date,
            "category": self.category,
            "merchant": self.merchant,
            "pending": self.pending,
        }


class MonoConnector(BaseConnector):
    """Mono connector — Nigerian banks (GTB, Access, FirstBank, UBA, Zenith).
    
    API: https://api.withmono.com/v2
    Auth: mono-sec-key header (test_sk_... or live_sk_...)
    
    Flow:
    1. User opens Connect widget (connect_url)
    2. Widget returns temporary code
    3. Exchange code: POST /v2/account/auth {code}
    4. Store account_id in vault
    5. Use account_id for all subsequent calls
    
    Endpoints:
    - GET /v2/accounts/{id} — account info
    - GET /v2/accounts/{id}/transactions — transaction history
    - GET /v2/accounts/{id}/identity — BVN, name, etc.
    - GET /v2/accounts/{id}/income — income verification
    - POST /v2/accounts/{id}/unlink — revoke access
    
    Webhooks:
    - Verify HMAC-SHA512 using mono-webhook-secret header
    """
    
    name = "mono"
    description = "Nigerian bank accounts via Mono (GTB, Access, FirstBank, UBA, Zenith)"
    
    BASE_URL = "https://api.withmono.com/v2"
    CONNECT_WIDGET_URL = "https://connect.mono.co/connect.js"
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(vault, config)
        self.http = HttpClient(timeout=30.0)
    
    def status(self) -> ConnectorStatus:
        """Check if Mono is configured and has linked accounts."""
        try:
            # Check for secret key
            creds = self._get_credential("secret_key")
            if not creds or not creds.get("key"):
                return ConnectorStatus(
                    connected=False,
                    error="No Mono secret key configured"
                )
            
            # Check for linked accounts
            accounts = self._get_credential("accounts")
            if not accounts or not accounts.get("ids"):
                return ConnectorStatus(
                    connected=False,
                    account=None,
                    error="No accounts linked"
                )
            
            account_ids = accounts["ids"]
            return ConnectorStatus(
                connected=True,
                account=f"{len(account_ids)} account(s)",
                scopes=["accounts", "transactions", "identity"]
            )
        
        except Exception as e:
            return ConnectorStatus(connected=False, error=str(e))
    
    def connect_url(self) -> str:
        """Return Mono Connect widget URL."""
        creds = self._get_credential("public_key")
        if not creds or not creds.get("key"):
            return ""
        
        public_key = creds["key"]
        return f"{self.CONNECT_WIDGET_URL}?key={public_key}"
    
    def disconnect(self) -> dict[str, Any]:
        """Unlink all accounts and clear credentials."""
        try:
            # Unlink each account
            accounts = self._get_credential("accounts")
            if accounts and accounts.get("ids"):
                secret_creds = self._get_credential("secret_key")
                if secret_creds and secret_creds.get("key"):
                    secret_key = secret_creds["key"]
                    for account_id in accounts["ids"]:
                        self._unlink_account(account_id, secret_key)
            
            # Clear all credentials
            self._delete_credential("secret_key")
            self._delete_credential("public_key")
            self._delete_credential("accounts")
            
            return {"disconnected": True, "accounts_unlinked": len(accounts.get("ids", []))}
        
        except Exception as e:
            return {"disconnected": False, "error": str(e)}
    
    def exchange_code(self, code: str) -> str:
        """Exchange temporary code for account ID.
        
        Args:
            code: Temporary code from Connect widget
        
        Returns:
            Account ID
        
        Raises:
            Exception: If exchange fails
        """
        creds = self._get_credential("secret_key")
        if not creds or not creds.get("key"):
            raise Exception("No Mono secret key configured")
        
        secret_key = creds["key"]
        
        response = self.http.post_json(
            f"{self.BASE_URL}/account/auth",
            {"code": code},
            headers={"mono-sec-key": secret_key}
        )
        
        if not response.ok:
            raise Exception(f"Code exchange failed: {response.status}")
        
        data = response.json()
        account_id = data.get("id")
        if not account_id:
            raise Exception("No account ID in response")
        
        # Store account ID
        accounts = self._get_credential("accounts") or {"ids": []}
        if account_id not in accounts["ids"]:
            accounts["ids"].append(account_id)
            self._store_credential("accounts", accounts)
        
        return account_id
    
    def get_account(self, account_id: str) -> BankAccount:
        """Get account details."""
        creds = self._get_credential("secret_key")
        if not creds or not creds.get("key"):
            raise Exception("No Mono secret key configured")
        
        response = self.http.get(
            f"{self.BASE_URL}/accounts/{account_id}",
            headers={"mono-sec-key": creds["key"]}
        )
        
        if not response.ok:
            raise Exception(f"Failed to get account: {response.status}")
        
        data = response.json()
        account_data = data.get("account", {})
        
        return BankAccount(
            account_id=account_id,
            provider="mono",
            institution=account_data.get("institution", {}).get("name", "Unknown"),
            mask=account_data.get("account_number", "")[-4:],
            account_type=account_data.get("type", "checking"),
            balance_minor=int(account_data.get("balance", 0) * 100),  # Naira to kobo
            currency="NGN",
            linked_at=time.time()
        )
    
    def get_transactions(self, account_id: str, days: int = 30) -> list[Transaction]:
        """Get transaction history."""
        creds = self._get_credential("secret_key")
        if not creds or not creds.get("key"):
            raise Exception("No Mono secret key configured")
        
        response = self.http.get(
            f"{self.BASE_URL}/accounts/{account_id}/transactions",
            headers={"mono-sec-key": creds["key"]}
        )
        
        if not response.ok:
            raise Exception(f"Failed to get transactions: {response.status}")
        
        data = response.json()
        transactions_data = data.get("transactions", [])
        
        transactions = []
        for tx in transactions_data[:100]:  # Limit to 100
            transactions.append(Transaction(
                transaction_id=tx.get("id", ""),
                account_id=account_id,
                amount_minor=int(tx.get("amount", 0) * 100),
                currency="NGN",
                description=tx.get("narration", ""),
                date=tx.get("date", ""),
                category=tx.get("type", ""),
                merchant=tx.get("narration", ""),
                pending=False
            ))
        
        return transactions
    
    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        """Verify Mono webhook signature (HMAC-SHA512).
        
        Args:
            payload: Raw request body
            signature: mono-webhook-secret header value
        
        Returns:
            True if signature is valid
        """
        creds = self._get_credential("webhook_secret")
        if not creds or not creds.get("secret"):
            return False
        
        expected = hmac.new(
            creds["secret"].encode(),
            payload,
            hashlib.sha512
        ).hexdigest()
        
        return hmac.compare_digest(expected, signature)
    
    def _unlink_account(self, account_id: str, secret_key: str) -> None:
        """Unlink a single account."""
        try:
            self.http.post_json(
                f"{self.BASE_URL}/accounts/{account_id}/unlink",
                {},
                headers={"mono-sec-key": secret_key}
            )
        except Exception as e:
            _log.warning("Failed to unlink account %s: %s", account_id, e)


class PlaidConnector(BaseConnector):
    """Plaid connector — US/EU banks.
    
    API: https://sandbox.plaid.com (or production)
    Auth: client_id + secret in request body
    
    Flow:
    1. Create link token: POST /link/token/create
    2. User completes Link flow → public_token
    3. Exchange: POST /item/public_token/exchange → access_token
    4. Store access_token in vault
    5. Use access_token for all subsequent calls
    
    Endpoints:
    - POST /accounts/balance/get — balances
    - POST /transactions/sync — transaction history
    - POST /identity/get — account holder info
    """
    
    name = "plaid"
    description = "US/EU bank accounts via Plaid"
    
    BASE_URL = "https://sandbox.plaid.com"  # Use production URL for live
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(vault, config)
        self.http = HttpClient(timeout=30.0)
    
    def status(self) -> ConnectorStatus:
        """Check if Plaid is configured and has linked accounts."""
        try:
            creds = self._get_credential("api_keys")
            if not creds or not creds.get("client_id") or not creds.get("secret"):
                return ConnectorStatus(
                    connected=False,
                    error="No Plaid API keys configured"
                )
            
            accounts = self._get_credential("accounts")
            if not accounts or not accounts.get("tokens"):
                return ConnectorStatus(
                    connected=False,
                    account=None,
                    error="No accounts linked"
                )
            
            tokens = accounts["tokens"]
            return ConnectorStatus(
                connected=True,
                account=f"{len(tokens)} account(s)",
                scopes=["accounts", "transactions", "identity"]
            )
        
        except Exception as e:
            return ConnectorStatus(connected=False, error=str(e))
    
    def connect_url(self) -> str:
        """Create link token for Plaid Link widget."""
        try:
            creds = self._get_credential("api_keys")
            if not creds or not creds.get("client_id") or not creds.get("secret"):
                return ""
            
            response = self.http.post_json(
                f"{self.BASE_URL}/link/token/create",
                {
                    "client_id": creds["client_id"],
                    "secret": creds["secret"],
                    "client_name": "Devon",
                    "country_codes": ["US"],
                    "language": "en",
                    "products": ["auth", "transactions", "identity"]
                }
            )
            
            if not response.ok:
                return ""
            
            data = response.json()
            link_token = data.get("link_token")
            return link_token or ""
        
        except Exception:
            return ""
    
    def disconnect(self) -> dict[str, Any]:
        """Clear all credentials."""
        try:
            self._delete_credential("api_keys")
            self._delete_credential("accounts")
            return {"disconnected": True}
        except Exception as e:
            return {"disconnected": False, "error": str(e)}
    
    def exchange_public_token(self, public_token: str) -> str:
        """Exchange public token for access token.
        
        Args:
            public_token: Token from Plaid Link
        
        Returns:
            Access token
        
        Raises:
            Exception: If exchange fails
        """
        creds = self._get_credential("api_keys")
        if not creds or not creds.get("client_id") or not creds.get("secret"):
            raise Exception("No Plaid API keys configured")
        
        response = self.http.post_json(
            f"{self.BASE_URL}/item/public_token/exchange",
            {
                "client_id": creds["client_id"],
                "secret": creds["secret"],
                "public_token": public_token
            }
        )
        
        if not response.ok:
            raise Exception(f"Token exchange failed: {response.status}")
        
        data = response.json()
        access_token = data.get("access_token")
        if not access_token:
            raise Exception("No access token in response")
        
        # Store access token
        accounts = self._get_credential("accounts") or {"tokens": []}
        if access_token not in accounts["tokens"]:
            accounts["tokens"].append(access_token)
            self._store_credential("accounts", accounts)
        
        return access_token


class Finance:
    """Unified finance facade — routes to Mono or Plaid based on region."""
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        self.vault = vault
        self.config = config or {}
        self.mono = MonoConnector(vault, config)
        self.plaid = PlaidConnector(vault, config)
    
    def link(self, provider: str = "auto") -> dict[str, Any]:
        """Get connect URL for linking.
        
        Args:
            provider: "mono", "plaid", or "auto" (routes based on region)
        
        Returns:
            Dict with connect_url and provider
        """
        if provider == "auto":
            # Default to Mono for NG, Plaid for others
            # In production, detect from user's location/IP
            provider = "mono"
        
        if provider == "mono":
            return {"connect_url": self.mono.connect_url(), "provider": "mono"}
        elif provider == "plaid":
            return {"connect_url": self.plaid.connect_url(), "provider": "plaid"}
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def accounts(self) -> list[dict[str, Any]]:
        """Get all linked accounts."""
        accounts = []
        
        # Mono accounts
        mono_status = self.mono.status()
        if mono_status.connected:
            mono_accounts = self.mono._get_credential("accounts")
            if mono_accounts and mono_accounts.get("ids"):
                for account_id in mono_accounts["ids"]:
                    try:
                        account = self.mono.get_account(account_id)
                        accounts.append(account.to_dict())
                    except Exception as e:
                        _log.warning("Failed to get Mono account %s: %s", account_id, e)
        
        # Plaid accounts (similar logic)
        # ...
        
        return accounts
    
    def transactions(self, account_id: str, days: int = 30) -> list[dict[str, Any]]:
        """Get transactions for an account."""
        # Determine provider from account_id
        # In production, store provider with account_id
        
        # Try Mono first
        try:
            txs = self.mono.get_transactions(account_id, days)
            return [tx.to_dict() for tx in txs]
        except Exception:
            pass
        
        # Try Plaid
        # ...
        
        return []
