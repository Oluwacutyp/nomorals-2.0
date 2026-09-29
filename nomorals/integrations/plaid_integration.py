"""Plaid integration for banking and finance.

Supports:
- Bank account balances
- Transaction history
- Recurring charges detection
- Liabilities (loans, credit cards)
- Investments

Usage:
    plaid = PlaidIntegration(account_manager)
    
    # Get balances
    balances = await plaid.get_balances(access_token="xxx")
    
    # Get transactions
    txns = await plaid.get_transactions(
        access_token="xxx", start_date="2026-09-01", end_date="2026-09-28"
    )
    
    # Get recurring charges
    recurring = await plaid.get_recurring_transactions(access_token="xxx")
    
    # Get liabilities
    liabilities = await plaid.get_liabilities(access_token="xxx")
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..core.logging_setup import get_logger

__all__ = ["PlaidIntegration", "Account", "Transaction", "Liability"]

_log = get_logger(__name__)

PLAID_API = "https://production.plaid.com"  # Use sandbox.plaid.com for testing


@dataclass
class Account:
    account_id: str
    name: str
    account_type: str  # depository, credit, investment, loan
    subtype: str = ""
    balance: float = 0.0
    currency: str = "USD"
    institution: str = ""
    mask: str = ""  # Last 4 digits
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id, "name": self.name,
            "type": self.account_type, "balance": self.balance,
            "currency": self.currency, "institution": self.institution,
        }


@dataclass
class Transaction:
    transaction_id: str
    account_id: str
    amount: float
    description: str
    date: str
    category: list[str] = field(default_factory=list)
    merchant: str = ""
    is_pending: bool = False
    payment_channel: str = ""  # in store, online, etc.
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id, "amount": self.amount,
            "description": self.description, "date": self.date,
            "category": self.category, "merchant": self.merchant,
        }


@dataclass
class Liability:
    liability_id: str
    account_id: str
    liability_type: str  # credit, student, mortgage
    balance: float = 0.0
    interest_rate: float = 0.0
    minimum_payment: float = 0.0
    due_date: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.liability_type, "balance": self.balance,
            "interest_rate": self.interest_rate, "minimum_payment": self.minimum_payment,
        }


class PlaidIntegration:
    """Plaid API integration for banking data."""
    
    def __init__(self, account_manager: AccountManager) -> None:
        self.account_manager = account_manager
        _log.info("Plaid integration initialized")
    
    async def _api_request(
        self, endpoint: str, data: dict[str, Any],
    ) -> dict[str, Any]:
        """Make Plaid API request."""
        # Get Plaid credentials
        cred = self.account_manager.get_credential("plaid", "default")
        
        data["client_id"] = cred.metadata.get("client_id", "")
        data["secret"] = cred.password
        
        url = f"{PLAID_API}/{endpoint}"
        body = json.dumps(data).encode()
        
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            _log.error(f"Plaid API error: {e.code} - {error_body}")
            raise
    
    # ── Balances ─────────────────────────────────────────────────────────────
    
    async def get_balances(self, access_token: str) -> list[Account]:
        """Get account balances."""
        result = await self._api_request("accounts/balance/get", {
            "access_token": access_token,
        })
        
        accounts = []
        for acc in result.get("accounts", []):
            balances = acc.get("balances", {})
            accounts.append(Account(
                account_id=acc.get("account_id", ""),
                name=acc.get("name", ""),
                account_type=acc.get("type", ""),
                subtype=acc.get("subtype", ""),
                balance=balances.get("current", 0.0) or 0.0,
                currency=balances.get("iso_currency_code", "USD"),
                mask=acc.get("mask", ""),
            ))
        
        return accounts
    
    # ── Transactions ─────────────────────────────────────────────────────────
    
    async def get_transactions(
        self, access_token: str, *,
        start_date: str, end_date: str, limit: int = 100,
    ) -> list[Transaction]:
        """Get transaction history."""
        result = await self._api_request("transactions/get", {
            "access_token": access_token,
            "start_date": start_date,
            "end_date": end_date,
            "options": {"count": limit},
        })
        
        transactions = []
        for txn in result.get("transactions", []):
            transactions.append(Transaction(
                transaction_id=txn.get("transaction_id", ""),
                account_id=txn.get("account_id", ""),
                amount=txn.get("amount", 0.0),
                description=txn.get("name", ""),
                date=txn.get("date", ""),
                category=txn.get("category", []),
                merchant=txn.get("merchant_name", ""),
                is_pending=txn.get("pending", False),
                payment_channel=txn.get("payment_channel", ""),
            ))
        
        return transactions
    
    async def get_recurring_transactions(self, access_token: str) -> list[dict[str, Any]]:
        """Detect recurring transactions."""
        result = await self._api_request("transactions/recurring/get", {
            "access_token": access_token,
        })
        
        recurring = []
        
        # Inflows (income)
        for stream in result.get("inflow_streams", []):
            recurring.append({
                "type": "income",
                "description": stream.get("description", ""),
                "average_amount": stream.get("average_amount", {}).get("amount", 0),
                "frequency": stream.get("frequency", ""),
                "last_date": stream.get("last_date", ""),
                "next_expected": stream.get("next_date", ""),
            })
        
        # Outflows (bills/subscriptions)
        for stream in result.get("outflow_streams", []):
            recurring.append({
                "type": "expense",
                "description": stream.get("description", ""),
                "average_amount": stream.get("average_amount", {}).get("amount", 0),
                "frequency": stream.get("frequency", ""),
                "last_date": stream.get("last_date", ""),
                "next_expected": stream.get("next_date", ""),
                "category": stream.get("categories", []),
            })
        
        return recurring
    
    # ── Liabilities ──────────────────────────────────────────────────────────
    
    async def get_liabilities(self, access_token: str) -> list[Liability]:
        """Get liabilities (credit cards, loans, mortgages)."""
        result = await self._api_request("liabilities/get", {
            "access_token": access_token,
        })
        
        liabilities = []
        liab_data = result.get("liabilities", {})
        
        # Credit cards
        for cc in liab_data.get("credit", []):
            liabilities.append(Liability(
                liability_id=cc.get("account_id", ""),
                account_id=cc.get("account_id", ""),
                liability_type="credit",
                balance=cc.get("last_statement_balance", 0.0) or 0.0,
                interest_rate=(cc.get("apr", [{}])[0].get("apr_percentage", 0.0) or 0.0) if cc.get("apr") else 0.0,
                minimum_payment=cc.get("minimum_payment_amount", 0.0) or 0.0,
                due_date=cc.get("last_payment_due_date", ""),
            ))
        
        # Student loans
        for loan in liab_data.get("student", []):
            liabilities.append(Liability(
                liability_id=loan.get("account_id", ""),
                account_id=loan.get("account_id", ""),
                liability_type="student_loan",
                balance=loan.get("outstanding_interest_amount", 0.0) or 0.0,
                interest_rate=loan.get("interest_rate_percentage", 0.0) or 0.0,
            ))
        
        # Mortgages
        for mort in liab_data.get("mortgage", []):
            liabilities.append(Liability(
                liability_id=mort.get("account_id", ""),
                account_id=mort.get("account_id", ""),
                liability_type="mortgage",
                balance=mort.get("current_late_fee", 0.0) or 0.0,
            ))
        
        return liabilities
    
    # ── Investments ──────────────────────────────────────────────────────────
    
    async def get_investments(self, access_token: str) -> dict[str, Any]:
        """Get investment holdings."""
        result = await self._api_request("investments/holdings/get", {
            "access_token": access_token,
        })
        
        holdings = []
        for holding in result.get("holdings", []):
            holdings.append({
                "security_id": holding.get("security_id", ""),
                "account_id": holding.get("account_id", ""),
                "quantity": holding.get("quantity", 0),
                "institution_price": holding.get("institution_price", 0.0),
                "institution_value": holding.get("institution_value", 0.0),
                "cost_basis": holding.get("cost_basis", 0.0),
            })
        
        securities = {}
        for sec in result.get("securities", []):
            securities[sec.get("security_id", "")] = {
                "name": sec.get("name", ""),
                "ticker": sec.get("ticker_symbol", ""),
                "type": sec.get("type", ""),
                "close_price": sec.get("close_price", 0.0),
            }
        
        return {"holdings": holdings, "securities": securities}
    
    # ── Identity ─────────────────────────────────────────────────────────────
    
    async def get_identity(self, access_token: str) -> list[dict[str, Any]]:
        """Get account holder identity info."""
        result = await self._api_request("identity/get", {
            "access_token": access_token,
        })
        
        identities = []
        for account in result.get("accounts", []):
            owners = account.get("owners", [])
            for owner in owners:
                names = owner.get("names", [])
                emails = [e["data"] for e in owner.get("emails", [])]
                phones = [p["data"] for p in owner.get("phone_numbers", [])]
                addresses = [a["data"] for a in owner.get("addresses", [])]
                
                identities.append({
                    "name": names[0] if names else "",
                    "emails": emails,
                    "phones": phones,
                    "addresses": addresses,
                })
        
        return identities
    
    # ── Link Token (for frontend auth flow) ──────────────────────────────────
    
    async def create_link_token(self, user_id: str) -> str:
        """Create a link token for Plaid Link frontend."""
        result = await self._api_request("link/token/create", {
            "user": {"client_user_id": user_id},
            "client_name": "NoMorals AI",
            "products": ["auth", "transactions", "balance", "investments"],
            "country_codes": ["US"],
            "language": "en",
        })
        
        return result.get("link_token", "")
    
    async def exchange_public_token(self, public_token: str) -> str:
        """Exchange a public token for an access token."""
        result = await self._api_request("item/public_token/exchange", {
            "public_token": public_token,
        })
        
        access_token = result.get("access_token", "")
        
        # Store in vault
        if access_token:
            self.account_manager.vault.store(
                service="plaid",
                username=f"access_{hash(access_token) % 10000}",
                password=access_token,
                credential_type="api_token",
                tags=["plaid", "banking"],
            )
        
        return access_token
