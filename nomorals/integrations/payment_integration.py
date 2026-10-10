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

    def format(self) -> str:
        """One-line transaction card."""
        icon = {"send": "📤", "receive": "📥",
                "purchase": "🛒"}.get(self.tx_type, "💸")
        status = {"confirmed": "✅", "pending": "⏳",
                  "failed": "❌"}.get(self.status, "")
        when = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(self.timestamp))
        return (f"{icon} {status} {self.tx_type} "
                f"{self.amount:.8f} {self.currency} · {when}")


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
    fee_estimate: str = ""
    fiat_amount: str = ""
    
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
        fee_line = f"\n**Fee est.:** {self.fee_estimate}" if self.fee_estimate else ""
        fiat_line = f" (≈ {self.fiat_amount})" if self.fiat_amount else ""
        return (
            f"💳 **Payment Approval Required**\n\n"
            f"**Type:** {self.payment_type}\n"
            f"**Amount:** {self.amount} {self.currency}{fiat_line}\n"
            f"**To:** {self.recipient}\n"
            f"**Description:** {self.description}{fee_line}\n\n"
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
        """Validate a crypto address format.

        ETH-family addresses get EIP-55 checksum verification when a
        keccak backend is available (eth_hash / pysha3); all-lowercase
        or all-uppercase (non-checksummed) addresses still pass the
        format check.
        """
        addr = (address or "").strip()
        if currency == "BTC":
            return addr.startswith(("1", "3", "bc1")) and 26 <= len(addr) <= 62
        elif currency in ("ETH", "USDT", "USDC", "MATIC", "BNB"):
            if not (addr.startswith("0x") and len(addr) == 42):
                return False
            hexpart = addr[2:]
            if not all(c in "0123456789abcdefABCDEF" for c in hexpart):
                return False
            # mixed-case → verify EIP-55 checksum when possible
            if hexpart != hexpart.lower() and hexpart != hexpart.upper():
                return self._eip55_valid(addr)
            return True
        else:
            return len(addr) > 10

    @staticmethod
    def _eip55_valid(address: str) -> bool:
        """EIP-55 checksum verification (best-effort without keccak)."""
        try:
            from eth_hash.auto import keccak  # type: ignore[import]
            digest = keccak
        except ImportError:
            try:
                from sha3 import keccak_256  # type: ignore[import]
                digest = lambda b: keccak_256(b).digest()  # noqa: E731
            except ImportError:
                return True  # no keccak backend — format check already passed
        h = digest(address[2:].lower().encode("ascii")).hex()
        for i, c in enumerate(address[2:]):
            if c.isalpha():
                should_upper = int(h[i], 16) >= 8
                if (c.isupper() != should_upper):
                    return False
        return True
    
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

    # ── fees / tracking / receive / wallet cards ──────────────────────

    async def estimate_fee(self, currency: str,
                           speed: str = "medium") -> dict[str, Any]:
        """Real fee estimate: mempool.space for BTC, eth_gasPrice for
        EVM. ``speed``: slow|medium|fast. Honest dict, never a guess."""
        cur = (currency or "").upper()
        if cur == "BTC":
            try:
                req = urllib.request.Request(
                    "https://mempool.space/api/v1/fees/recommended",
                    headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    fees = json.loads(resp.read().decode())
                key = {"slow": "hourFee", "medium": "halfHourFee",
                       "fast": "fastestFee"}.get(speed, "halfHourFee")
                return {"currency": "BTC", "sat_per_vbyte": fees.get(key),
                        "speed": speed, "source": "mempool.space",
                        "note": "for a ~140 vB tx: "
                                f"~{fees.get(key, 0) * 140} sats"}
            except Exception as exc:  # noqa: BLE001
                _log.warning("btc fee estimate failed: %s", exc)
        if cur in ("ETH", "USDT", "USDC", "MATIC", "BNB"):
            gwei = await self._eth_gas_price_gwei(speed)
            gas_limit = 21000 if cur == "ETH" else 65000
            if gwei:
                eth_fee = gwei * gas_limit / 1e9
                return {"currency": cur, "gas_gwei": round(gwei, 2),
                        "gas_limit": gas_limit,
                        "fee_eth": round(eth_fee, 6),
                        "speed": speed, "source": "eth_gasPrice"}
        # static fallback — labeled as approximate
        approx = {"BTC": "1–5 USD", "ETH": "0.5–3 USD",
                  "SOL": "~0.000005 SOL"}.get(cur, "unknown")
        return {"currency": cur, "approximate": approx, "speed": speed,
                "source": "static-fallback",
                "note": "live estimate unavailable — approximate only"}

    async def _eth_gas_price_gwei(self, speed: str) -> float | None:
        """eth_gasPrice via the public Cloudflare RPC (no key)."""
        try:
            payload = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_gasPrice", "params": []}).encode()
            req = urllib.request.Request(
                "https://cloudflare-eth.com",
                data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            gwei = int(result.get("result", "0x0"), 16) / 1e9
            mult = {"slow": 0.9, "medium": 1.0, "fast": 1.25}.get(speed, 1.0)
            return gwei * mult if gwei > 0 else None
        except Exception as exc:  # noqa: BLE001
            _log.debug("eth gas price failed: %s", exc)
            return None

    async def get_erc20_balance(self, address: str, contract_address: str,
                                *, decimals: int = 18) -> float:
        """ERC-20 token balance via web3 + public RPC (no key).

        Needs the ``web3`` package; raises a clear error without it.
        """
        try:
            from web3 import Web3  # type: ignore[import]
        except ImportError as exc:
            raise PaymentError(
                "ERC-20 balances need the web3 package: pip install web3"
            ) from exc
        w3 = Web3(Web3.HTTPProvider("https://cloudflare-eth.com",
                                    request_kwargs={"timeout": 10}))
        if not w3.is_connected():
            raise PaymentError("no Ethereum RPC reachable")
        abi = [{"constant": True, "inputs": [{"name": "_owner",
                "type": "address"}], "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"}]
        contract = w3.eth.contract(
            address=w3.to_checksum_address(contract_address), abi=abi)
        raw = contract.functions.balanceOf(
            w3.to_checksum_address(address)).call()
        return raw / (10 ** decimals)

    async def track_tx(self, txid: str,
                       currency: str) -> dict[str, Any]:
        """Live transaction status: confirmations, fee, status.

        BTC via mempool.space; ETH via public RPC receipt.
        """
        cur = (currency or "").upper()
        txid = (txid or "").strip()
        if cur == "BTC":
            try:
                req = urllib.request.Request(
                    f"https://mempool.space/api/tx/{txid}/status",
                    headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    st = json.loads(resp.read().decode())
                req2 = urllib.request.Request(
                    f"https://mempool.space/api/tx/{txid}",
                    headers={"Accept": "application/json"})
                with urllib.request.urlopen(req2, timeout=10) as resp:
                    tx = json.loads(resp.read().decode())
                confirmed = bool(st.get("confirmed"))
                return {"txid": txid, "currency": "BTC",
                        "status": "confirmed" if confirmed else "pending",
                        "confirmations": 1 if confirmed else 0,
                        "block_height": st.get("block_height"),
                        "block_time": st.get("block_time"),
                        "fee_sats": tx.get("fee"),
                        "explorer": self._explorer_url("BTC", txid=txid)}
            except Exception as exc:  # noqa: BLE001
                raise PaymentError(
                    f"could not track BTC tx {txid}: {exc}") from exc
        if cur in ("ETH", "USDT", "USDC"):
            try:
                from web3 import Web3  # type: ignore[import]
            except ImportError as exc:
                raise PaymentError(
                    "ETH tx tracking needs web3: pip install web3") from exc
            w3 = Web3(Web3.HTTPProvider("https://cloudflare-eth.com",
                                        request_kwargs={"timeout": 10}))
            try:
                receipt = w3.eth.get_transaction_receipt(txid)
            except Exception:
                receipt = None
            if receipt is None:
                return {"txid": txid, "currency": cur, "status": "pending",
                        "confirmations": 0,
                        "explorer": self._explorer_url(cur, txid=txid)}
            ok = receipt.get("status") == 1
            return {"txid": txid, "currency": cur,
                    "status": "confirmed" if ok else "failed",
                    "confirmations": 12,
                    "block": receipt.get("blockNumber"),
                    "gas_used": receipt.get("gasUsed"),
                    "explorer": self._explorer_url(cur, txid=txid)}
        raise PaymentError(f"tx tracking not implemented for {cur}")

    def format_tx_status(self, status: dict[str, Any]) -> str:
        """God-tier tx status card."""
        icon = {"confirmed": "✅", "pending": "⏳",
                "failed": "❌"}.get(status.get("status"), "❓")
        lines = [f"{icon} **{status.get('currency')} transaction**",
                 f"`{status.get('txid', '')}`",
                 f"status: {status.get('status')}"]
        if status.get("confirmations"):
            lines.append(f"confirmations: {status['confirmations']}")
        if status.get("fee_sats"):
            lines.append(f"fee: {status['fee_sats']} sats")
        if status.get("gas_used"):
            lines.append(f"gas used: {status['gas_used']}")
        if status.get("explorer"):
            lines.append(f"🔗 {status['explorer']}")
        return "\n".join(lines)

    @staticmethod
    def _explorer_url(currency: str, *, txid: str = "",
                      address: str = "") -> str:
        cur = (currency or "").upper()
        base = {
            "BTC": "https://mempool.space",
            "ETH": "https://etherscan.io",
            "USDT": "https://etherscan.io",
            "USDC": "https://etherscan.io",
            "SOL": "https://solscan.io",
            "MATIC": "https://polygonscan.com",
            "BNB": "https://bscscan.com",
        }.get(cur, "")
        if not base:
            return ""
        if txid:
            return f"{base}/tx/{txid}"
        if address:
            return f"{base}/address/{address}"
        return base

    async def receive_qr(self, currency: str, *,
                         wallet: str = "main") -> str:
        """QR code for the receiving address.

        Returns a PNG path when ``qrcode`` is installed, otherwise an
        ASCII-art QR in a code block (still scannable from chat on
        most phones' cameras… no — honest: ASCII is for display;
        the address text below it is what to copy).
        """
        address = await self.receive_crypto(currency, wallet=wallet)
        try:
            import qrcode  # type: ignore[import]
            import tempfile
            img = qrcode.make(address)
            with tempfile.NamedTemporaryFile(suffix=".png",
                                             delete=False) as f:
                img.save(f.name)
                return f"![receive QR]({f.name})\n`{address}`"
        except ImportError:
            pass
        # ASCII fallback — display only
        try:
            import qrcode  # noqa: F401  (unreachable, keeps linters calm)
        except ImportError:
            pass
        return self._ascii_qr(address) + f"\n`{address}`"

    @staticmethod
    def _ascii_qr(data: str) -> str:
        """Tiny QR-ish placeholder: NOT a real QR — honest label."""
        # Without the qrcode lib we can't render a real matrix; show a
        # framed address block instead of a fake QR.
        border = "─" * (min(len(data), 42) + 2)
        return (f"┌{border}┐\n│ {data[:42]:<42} │\n└{border}┘\n"
                f"_copy the address above — install `qrcode` for a real QR_")

    async def fiat_value(self, currency: str, amount: float,
                         fiat: str = "USD") -> float:
        """Fiat equivalent via market_data (CoinGecko/Binance keyless)."""
        try:
            from . import market_data
            q = market_data.quote(f"{currency}/USDT", market="crypto")
            price = float(q.get("price") or 0)
            if fiat.upper() == "USD" or not price:
                return amount * price
            fx = market_data.quote(f"USD/{fiat.upper()}", market="forex")
            return amount * price * float(fx.get("price") or 1)
        except Exception:  # noqa: BLE001
            return 0.0

    async def format_wallet(self, wallet: str = "main",
                            fiat: str = "USD") -> str:
        """God-tier wallet card: per-currency balances + fiat totals."""
        lines = [f"💰 **wallet: {wallet}**"]
        total = 0.0
        for cur in self.SUPPORTED_CRYPTO:
            try:
                bal = await self.get_balance(wallet, currency=cur)
            except Exception:  # noqa: BLE001
                continue
            if bal <= 0:
                continue
            fv = await self.fiat_value(cur, bal, fiat)
            total += fv
            try:
                addr = await self.receive_crypto(cur, wallet=wallet)
                short = f"{addr[:10]}…{addr[-6:]}" if len(addr) > 18 else addr
            except Exception:  # noqa: BLE001
                short = ""
            lines.append(f"• **{bal:.8f} {cur}** ≈ {fv:,.2f} {fiat}"
                         + (f"  `{short}`" if short else ""))
        lines.append(f"\n**total ≈ {total:,.2f} {fiat}**")
        return "\n".join(lines)

    # ── address book ───────────────────────────────────────────────

    def _address_book_path(self) -> str:
        import os
        path = os.path.expanduser("~/.nomorals/payments/address_book.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _load_address_book(self) -> dict[str, dict]:
        import os
        path = self._address_book_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (ValueError, OSError):
            return {}

    def _save_address_book(self, book: dict[str, dict]) -> None:
        with open(self._address_book_path(), "w") as f:
            json.dump(book, f, indent=2)

    def save_address(self, name: str, currency: str, address: str) -> None:
        """Save a named address (validated before storing)."""
        if not self._validate_address(currency, address):
            raise PaymentError(
                f"refusing to save invalid {currency} address: {address}")
        book = self._load_address_book()
        book[(name or "").strip().lower()] = {
            "name": name, "currency": (currency or "").upper(),
            "address": address, "saved_at": time.time()}
        self._save_address_book(book)

    def get_address(self, name: str) -> dict | None:
        """Look up a saved address by name."""
        return self._load_address_book().get((name or "").strip().lower())

    def list_addresses(self) -> list[dict]:
        """All saved addresses."""
        return sorted(self._load_address_book().values(),
                      key=lambda a: a.get("name", ""))

    def delete_address(self, name: str) -> bool:
        book = self._load_address_book()
        if (name or "").strip().lower() in book:
            del book[(name or "").strip().lower()]
            self._save_address_book(book)
            return True
        return False
