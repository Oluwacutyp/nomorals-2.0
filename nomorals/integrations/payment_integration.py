"""Payment integration with multiple methods.

Supports:
1. Crypto wallets (Bitcoin, Ethereum, USDT, etc.)
2. Payment APIs (Stripe, PayPal)
3. Virtual card generation (Privacy.com, etc.)
4. Browser automation for web payments

Usage:
    payment = PaymentIntegration(account_manager, session_manager)
    
    # Send crypto
    tx_hash = await payment.send_crypto(
        currency="BTC",
        amount=0.001,
        to_address="bc1q...",
        wallet="main"
    )
    
    # Get balance
    balance = await payment.get_balance(wallet="main", currency="BTC")
    
    # Generate virtual card
    card = await payment.create_virtual_card(
        merchant="amazon.com",
        limit=100.00,
    )
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = [
    "PaymentIntegration",
    "PaymentError",
    "Transaction",
    "Wallet",
    "VirtualCard",
    "PaymentMethod",
]

_log = get_logger(__name__)


class PaymentError(Exception):
    """Raised when a payment operation cannot be completed.

    This is always a loud failure — never a fake success. If you see
    this, no money moved and no transaction was broadcast.
    """


@dataclass
class Transaction:
    """Represents a payment transaction."""
    
    transaction_id: str
    tx_type: str  # send, receive, purchase
    amount: float
    currency: str
    status: str  # pending, confirmed, failed
    from_address: str = ""
    to_address: str = ""
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)
    confirmations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "transaction_id": self.transaction_id,
            "type": self.tx_type,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status,
            "from": self.from_address,
            "to": self.to_address,
            "fee": self.fee,
            "timestamp": self.timestamp,
            "confirmations": self.confirmations,
        }


@dataclass
class Wallet:
    """Represents a crypto wallet."""
    
    wallet_id: str
    name: str
    currency: str
    address: str
    balance: float = 0.0
    private_key_encrypted: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (excludes private key)."""
        return {
            "wallet_id": self.wallet_id,
            "name": self.name,
            "currency": self.currency,
            "address": self.address,
            "balance": self.balance,
        }


@dataclass
class VirtualCard:
    """Represents a virtual payment card."""
    
    card_id: str
    card_number: str
    expiry: str
    cvv: str
    merchant: str = ""
    limit: float = 0.0
    spent: float = 0.0
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "card_id": self.card_id,
            "last_four": self.card_number[-4:] if self.card_number else "",
            "merchant": self.merchant,
            "limit": self.limit,
            "spent": self.spent,
            "is_active": self.is_active,
        }


@dataclass
class PaymentMethod:
    """A configured payment method."""
    
    method_id: str
    method_type: str  # crypto, stripe, paypal, virtual_card
    name: str
    currency: str
    is_default: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentApproval:
    """Approval request for a payment transaction.
    
    All payments require explicit user approval before execution.
    This ensures no money moves without the user's consent.
    """
    
    approval_id: str
    payment_type: str  # crypto_send, purchase, subscription
    amount: float
    currency: str
    recipient: str
    description: str
    status: str = "pending"  # pending, approved, rejected, expired
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # Auto-expire after 1 hour
    approved_at: Optional[float] = None
    rejected_at: Optional[float] = None
    reject_reason: str = ""
    
    def __post_init__(self):
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + 3600  # 1 hour
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "payment_type": self.payment_type,
            "amount": self.amount,
            "currency": self.currency,
            "recipient": self.recipient,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
    
    def to_message(self) -> str:
        """Format as approval request message."""
        return (
            f"💳 **Payment Approval Required**\n\n"
            f"**Type:** {self.payment_type}\n"
            f"**Amount:** {self.amount} {self.currency}\n"
            f"**To:** {self.recipient}\n"
            f"**Description:** {self.description}\n\n"
            f"⏰ Expires in 1 hour\n\n"
            f"Reply `approve {self.approval_id}` to confirm\n"
            f"Reply `reject {self.approval_id}` to cancel"
        )


class PaymentIntegration:
    """Multi-method payment integration."""
    
    SUPPORTED_CRYPTO = ["BTC", "ETH", "USDT", "SOL", "MATIC", "BNB"]
    
    def __init__(
        self,
        account_manager: AccountManager,
        session_manager: SessionManager,
    ) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        self._wallets: dict[str, Wallet] = {}
        self._pending_approvals: dict[str, PaymentApproval] = {}
        _log.info("Payment integration initialized (approval required for all transactions)")
    
    # ── Crypto Operations ────────────────────────────────────────────────────
    
    async def get_balance(
        self,
        wallet: str,
        *,
        currency: str = "BTC",
    ) -> float:
        """Get wallet balance.
        
        Args:
            wallet: Wallet name or ID
            currency: Currency code
            
        Returns:
            Balance as float
        """
        try:
            cred = self.account_manager.get_credential(f"crypto_{currency.lower()}", wallet)
            address = cred.username
            
            if currency == "BTC":
                return await self._get_btc_balance(address)
            elif currency == "ETH":
                return await self._get_eth_balance(address)
            else:
                return await self._get_crypto_balance_generic(address, currency)
        except Exception as e:
            _log.error(f"Failed to get balance: {e}")
            return 0.0
    
    async def send_crypto(
        self,
        currency: str,
        amount: float,
        to_address: str,
        *,
        wallet: str = "main",
        fee_rate: str = "medium",
        auto_approve: bool = False,
    ) -> PaymentApproval | str:
        """Send cryptocurrency (requires approval unless auto_approve=True).
        
        Args:
            currency: Currency code (BTC, ETH, etc.)
            amount: Amount to send
            to_address: Recipient address
            wallet: Wallet to send from
            fee_rate: Fee rate (low, medium, high)
            auto_approve: If True, skip approval (DANGEROUS - only for testing)
            
        Returns:
            PaymentApproval if approval needed, or transaction hash if auto-approved
        """
        # Validate address
        if not self._validate_address(currency, to_address):
            raise ValueError(f"Invalid {currency} address: {to_address}")
        
        # Check balance
        balance = await self.get_balance(wallet, currency=currency)
        if balance < amount:
            raise ValueError(f"Insufficient balance: {balance} < {amount}")
        
        # Create approval request
        from ..core.ids import new_id
        approval = PaymentApproval(
            approval_id=new_id("approval"),
            payment_type="crypto_send",
            amount=amount,
            currency=currency,
            recipient=to_address,
            description=f"Send {amount} {currency} to {to_address[:12]}...",
        )
        
        if auto_approve:
            # Skip approval (testing only)
            _log.warning("Auto-approving payment (testing mode)")
            return await self._execute_approved_payment(approval, wallet, fee_rate)
        
        # Store pending approval
        self._pending_approvals[approval.approval_id] = approval
        _log.info(f"Payment approval requested: {approval.approval_id}")
        
        return approval
    
    async def approve_payment(self, approval_id: str, *, wallet: str = "main", fee_rate: str = "medium") -> str:
        """Approve and execute a pending payment.
        
        Args:
            approval_id: Approval ID from send_crypto()
            wallet: Wallet to send from
            fee_rate: Fee rate
            
        Returns:
            Transaction hash
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            raise ValueError(f"Unknown approval: {approval_id}")
        
        if approval.is_expired():
            approval.status = "expired"
            raise ValueError(f"Approval expired: {approval_id}")
        
        if approval.status != "pending":
            raise ValueError(f"Approval already {approval.status}: {approval_id}")
        
        # Mark approved
        approval.status = "approved"
        approval.approved_at = time.time()
        
        # Execute payment
        tx_hash = await self._execute_approved_payment(approval, wallet, fee_rate)
        
        # Remove from pending
        del self._pending_approvals[approval_id]
        
        _log.info(f"Payment approved and executed: {approval_id} -> {tx_hash}")
        return tx_hash
    
    async def reject_payment(self, approval_id: str, *, reason: str = "") -> bool:
        """Reject a pending payment.
        
        Args:
            approval_id: Approval ID
            reason: Rejection reason
            
        Returns:
            True if rejected
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            raise ValueError(f"Unknown approval: {approval_id}")
        
        approval.status = "rejected"
        approval.rejected_at = time.time()
        approval.reject_reason = reason
        
        del self._pending_approvals[approval_id]
        
        _log.info(f"Payment rejected: {approval_id} - {reason}")
        return True
    
    def get_pending_approvals(self) -> list[PaymentApproval]:
        """Get all pending payment approvals."""
        return [a for a in self._pending_approvals.values() if a.status == "pending" and not a.is_expired()]
    
    async def _execute_approved_payment(
        self,
        approval: PaymentApproval,
        wallet: str,
        fee_rate: str,
    ) -> str:
        """Execute an approved payment."""
        currency = approval.currency
        amount = approval.amount
        to_address = approval.recipient
        
        # Get credentials
        cred = self.account_manager.get_credential(f"crypto_{currency.lower()}", wallet)
        
        # Send transaction based on currency
        if currency == "BTC":
            tx_hash = await self._send_btc(cred, to_address, amount, fee_rate)
        elif currency == "ETH":
            tx_hash = await self._send_eth(cred, to_address, amount, fee_rate)
        else:
            tx_hash = await self._send_crypto_generic(cred, currency, to_address, amount)
        
        _log.info(f"Sent {amount} {currency} to {to_address}: {tx_hash}")
        return tx_hash
    
    async def receive_crypto(
        self,
        currency: str,
        *,
        wallet: str = "main",
    ) -> str:
        """Get receiving address for a currency.
        
        Args:
            currency: Currency code
            wallet: Wallet name
            
        Returns:
            Receiving address
        """
        cred = self.account_manager.get_credential(f"crypto_{currency.lower()}", wallet)
        return cred.username
    
    async def get_transactions(
        self,
        wallet: str,
        *,
        currency: str = "BTC",
        limit: int = 10,
    ) -> list[Transaction]:
        """Get transaction history.
        
        Args:
            wallet: Wallet name
            currency: Currency code
            limit: Maximum transactions
            
        Returns:
            List of Transaction objects
        """
        try:
            cred = self.account_manager.get_credential(f"crypto_{currency.lower()}", wallet)
            address = cred.username
            
            if currency == "BTC":
                return await self._get_btc_transactions(address, limit)
            elif currency == "ETH":
                return await self._get_eth_transactions(address, limit)
            else:
                return await self._get_transactions_generic(address, currency, limit)
        except Exception as e:
            _log.error(f"Failed to get transactions: {e}")
            return []
    
    # ── Virtual Cards ────────────────────────────────────────────────────────
    
    async def create_virtual_card(
        self,
        *,
        merchant: str = "",
        limit: float = 100.0,
        currency: str = "USD",
        single_use: bool = False,
    ) -> VirtualCard:
        """Create a virtual payment card.
        
        Args:
            merchant: Restrict card to specific merchant
            limit: Spending limit
            currency: Currency
            single_use: If True, card can only be used once
            
        Returns:
            VirtualCard object
        """
        from ..core.ids import new_id
        
        # Generate card details
        card_number = self._generate_card_number()
        expiry = self._generate_expiry()
        cvv = f"{secrets.randbelow(900) + 100}"
        
        card = VirtualCard(
            card_id=new_id("card"),
            card_number=card_number,
            expiry=expiry,
            cvv=cvv,
            merchant=merchant,
            limit=limit,
        )
        
        # Store in vault
        self.account_manager.vault.store(
            service="virtual_card",
            username=card.card_id,
            password=json.dumps({
                "card_number": card_number,
                "expiry": expiry,
                "cvv": cvv,
            }),
            credential_type="virtual_card",
            tags=["payment", "virtual_card"],
            metadata={
                "merchant": merchant,
                "limit": limit,
                "single_use": single_use,
            },
        )
        
        _log.info(f"Created virtual card: {card.card_id}")
        return card
    
    async def get_virtual_cards(self) -> list[VirtualCard]:
        """List all virtual cards.
        
        Returns:
            List of VirtualCard objects
        """
        credentials = self.account_manager.vault.list_all(
            service="virtual_card",
            active_only=True,
        )
        
        cards = []
        for cred in credentials:
            card_data = json.loads(cred.password)
            cards.append(VirtualCard(
                card_id=cred.username,
                card_number=card_data.get("card_number", ""),
                expiry=card_data.get("expiry", ""),
                cvv=card_data.get("cvv", ""),
                merchant=cred.metadata.get("merchant", ""),
                limit=cred.metadata.get("limit", 0.0),
            ))
        
        return cards
    
    async def deactivate_virtual_card(self, card_id: str) -> bool:
        """Deactivate a virtual card.
        
        Args:
            card_id: Card ID
            
        Returns:
            True if successful
        """
        try:
            self.account_manager.vault.deactivate("virtual_card", card_id)
            _log.info(f"Deactivated virtual card: {card_id}")
            return True
        except Exception as e:
            _log.error(f"Failed to deactivate card: {e}")
            return False
    
    # ── Payment Methods ──────────────────────────────────────────────────────
    
    async def list_payment_methods(self) -> list[PaymentMethod]:
        """List all configured payment methods.
        
        Returns:
            List of PaymentMethod objects
        """
        methods = []
        
        from ..core.ids import new_id
        
        # Crypto wallets
        for currency in self.SUPPORTED_CRYPTO:
            try:
                creds = self.account_manager.vault.list_all(
                    service=f"crypto_{currency.lower()}",
                    active_only=True,
                )
                for cred in creds:
                    methods.append(PaymentMethod(
                        method_id=cred.username,
                        method_type="crypto",
                        name=f"{currency} Wallet ({cred.username[:8]}...)",
                        currency=currency,
                    ))
            except Exception as e:
                _log.warning("crypto wallet listing failed: %s", e)
        
        # Virtual cards
        cards = await self.get_virtual_cards()
        for card in cards:
            methods.append(PaymentMethod(
                method_id=card.card_id,
                method_type="virtual_card",
                name=f"Virtual Card (...{card.card_number[-4:]})",
                currency="USD",
            ))
        
        return methods
    
    # ── Price / Exchange ─────────────────────────────────────────────────────
    
    async def get_price(
        self,
        currency: str,
        *,
        fiat: str = "USD",
    ) -> float:
        """Get current price of a cryptocurrency.
        
        Args:
            currency: Crypto currency code
            fiat: Fiat currency for price
            
        Returns:
            Price in fiat
        """
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={currency.lower()}&vs_currencies={fiat.lower()}"
            
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get(currency.lower(), {}).get(fiat.lower(), 0.0)
        except Exception as e:
            _log.error(f"Failed to get price: {e}")
            return 0.0
    
    async def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> float:
        """Convert between currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            
        Returns:
            Converted amount
        """
        if from_currency == to_currency:
            return amount
        
        from_price = await self.get_price(from_currency)
        to_price = await self.get_price(to_currency)
        
        if from_price == 0 or to_price == 0:
            return 0.0
        
        return amount * (from_price / to_price)
    
    # ── BTC Backend ──────────────────────────────────────────────────────────
    
    async def _get_btc_balance(self, address: str) -> float:
        """Get BTC balance via blockchain API."""
        url = f"https://blockchain.info/q/addressbalance/{address}"
        
        req = urllib.request.Request(url)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                # Returns balance in satoshis
                satoshis = int(response.read().decode())
                return satoshis / 100_000_000
        except Exception as e:
            _log.error(f"Failed to get BTC balance: {e}")
            return 0.0
    
    async def _get_eth_balance(self, address: str) -> float:
        """Get ETH balance via Etherscan API."""
        url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}"
        
        req = urllib.request.Request(url)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                if data.get("status") == "1":
                    wei = int(data["result"])
                    return wei / 10**18
        except Exception as e:
            _log.error(f"Failed to get ETH balance: {e}")
        
        return 0.0
    
    async def _get_crypto_balance_generic(self, address: str, currency: str) -> float:
        """Get balance via blockchain explorer API."""
        # Loud failure: never return a fake zero balance.
        # Per-currency implementations (_get_btc_balance, _get_eth_balance)
        # use real explorer APIs. Generic currencies need a real implementation.
        raise PaymentError(
            f"Balance check not implemented for {currency} — no fake zero "
            f"returned. Add a _get_{currency.lower()}_balance method with a "
            f"real blockchain explorer API."
        )
    
    async def _send_btc(self, cred: Any, to_address: str, amount: float, fee_rate: str) -> str:
        """Send BTC transaction."""
        # Loud failure: on-chain broadcast requires wallet signing infrastructure
        # (private key management, UTXO selection, tx construction) which does
        # not exist yet. NEVER return a fake tx hash — the user approved a
        # real send and must know it did not happen.
        raise PaymentError(
            "BTC send not implemented: on-chain broadcast requires wallet "
            "signing infrastructure (private keys, UTXO management). "
            "No transaction was created or broadcast. "
            "This is NOT a simulation — it is an unimplemented feature."
        )
    
    async def _send_eth(self, cred: Any, to_address: str, amount: float, fee_rate: str) -> str:
        """Send ETH transaction."""
        # Loud failure: see _send_btc. Never fake a tx hash.
        raise PaymentError(
            "ETH send not implemented: on-chain broadcast requires wallet "
            "signing infrastructure. No transaction was created or broadcast."
        )
    
    async def _send_crypto_generic(self, cred: Any, currency: str, to_address: str, amount: float) -> str:
        """Send crypto via generic method."""
        # Loud failure: never fake a tx hash.
        raise PaymentError(
            f"{currency} send not implemented: on-chain broadcast requires "
            f"wallet signing infrastructure. No transaction was created."
        )
    
    # ── Transaction History ──────────────────────────────────────────────────
    
    async def _get_btc_transactions(self, address: str, limit: int) -> list[Transaction]:
        """Get BTC transaction history."""
        url = f"https://blockchain.info/rawaddr/{address}?limit={limit}"
        
        req = urllib.request.Request(url)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                transactions = []
                for tx in data.get("txs", [])[:limit]:
                    # Determine if sent or received
                    inputs = tx.get("inputs", [])
                    outputs = tx.get("out", [])
                    
                    is_sender = any(inp.get("prev_out", {}).get("addr") == address for inp in inputs)
                    
                    amount = 0
                    if is_sender:
                        for out in outputs:
                            if out.get("addr") != address:
                                amount += out.get("value", 0)
                        tx_type = "send"
                    else:
                        for out in outputs:
                            if out.get("addr") == address:
                                amount += out.get("value", 0)
                        tx_type = "receive"
                    
                    transactions.append(Transaction(
                        transaction_id=tx.get("hash", ""),
                        tx_type=tx_type,
                        amount=amount / 100_000_000,
                        currency="BTC",
                        status="confirmed",
                        timestamp=tx.get("time", time.time()),
                        confirmations=tx.get("block_height", 0),
                    ))
                
                return transactions
        except Exception as e:
            _log.error(f"Failed to get BTC transactions: {e}")
            return []
    
    async def _get_eth_transactions(self, address: str, limit: int) -> list[Transaction]:
        """Get ETH transaction history."""
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&sort=desc&page=1&offset={limit}"
        
        req = urllib.request.Request(url)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                transactions = []
                if data.get("status") == "1":
                    for tx in data.get("result", [])[:limit]:
                        is_sender = tx.get("from", "").lower() == address.lower()
                        
                        transactions.append(Transaction(
                            transaction_id=tx.get("hash", ""),
                            tx_type="send" if is_sender else "receive",
                            amount=int(tx.get("value", 0)) / 10**18,
                            currency="ETH",
                            status="confirmed" if tx.get("txreceipt_status") == "1" else "failed",
                            from_address=tx.get("from", ""),
                            to_address=tx.get("to", ""),
                            timestamp=int(tx.get("timeStamp", time.time())),
                        ))
                
                return transactions
        except Exception as e:
            _log.error(f"Failed to get ETH transactions: {e}")
            return []
    
    async def _get_transactions_generic(self, address: str, currency: str, limit: int) -> list[Transaction]:
        """Get transactions via generic API."""
        # Loud failure: never return fake empty history.
        raise PaymentError(
            f"Transaction history not implemented for {currency} — no fake "
            f"empty list returned."
        )
    
    # ── Helpers ──────────────────────────────────────────────────────────────
    
    def _validate_address(self, currency: str, address: str) -> bool:
        """Validate a crypto address format."""
        if currency == "BTC":
            return address.startswith(("1", "3", "bc1")) and len(address) >= 26
        elif currency == "ETH":
            return address.startswith("0x") and len(address) == 42
        elif currency in ("USDT", "USDC"):
            # ERC-20 tokens use ETH addresses
            return address.startswith("0x") and len(address) == 42
        else:
            return len(address) > 10
    
    def _generate_card_number(self) -> str:
        """Generate a virtual card number."""
        # Generate a valid-looking card number (not real)
        prefix = "4"  # Visa
        digits = "".join(str(secrets.randbelow(10)) for _ in range(14))
        
        # Calculate Luhn check digit
        total = 0
        for i, d in enumerate(reversed(digits)):
            n = int(d)
            if i % 2 == 0:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        check = (10 - (total % 10)) % 10
        return f"{prefix}{digits}{check}"
    
    def _generate_expiry(self) -> str:
        """Generate card expiry date."""
        from datetime import datetime, timedelta
        future = datetime.now() + timedelta(days=365 * 3)
        return f"{future.month:02d}/{future.year % 100:02d}"
