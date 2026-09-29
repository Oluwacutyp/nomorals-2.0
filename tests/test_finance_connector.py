"""Tests for finance connector — Mono + Plaid with mocked HTTP."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from nomorals.connectors.finance import (
    BankAccount,
    Finance,
    MonoConnector,
    PlaidConnector,
    Transaction,
)
from nomorals.connectors.vault import CredentialVault


class TestMonoConnector:
    """Test Mono connector with mocked HTTP."""
    
    def test_mono_status_no_key(self, tmp_path):
        """Mono status returns disconnected when no key configured."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        status = connector.status()
        assert not status.connected
        assert "No Mono secret key" in status.error
    
    def test_mono_status_with_key_no_accounts(self, tmp_path):
        """Mono status returns disconnected when key set but no accounts."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        # Store secret key
        vault.store("mono", "secret_key", {"key": "test_sk_123"})
        
        status = connector.status()
        assert not status.connected
        assert "No accounts linked" in status.error
    
    def test_mono_status_connected(self, tmp_path):
        """Mono status returns connected when key and accounts configured."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        # Store credentials
        vault.store("mono", "secret_key", {"key": "test_sk_123"})
        vault.store("mono", "accounts", {"ids": ["acc_123", "acc_456"]})
        
        status = connector.status()
        assert status.connected
        assert "2 account(s)" in status.account
    
    def test_mono_exchange_code_success(self, tmp_path):
        """Mono exchange_code returns account ID on success."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        # Store secret key
        vault.store("mono", "secret_key", {"key": "test_sk_123"})
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "acc_new_123"}
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            account_id = connector.exchange_code("temp_code_abc")
        
        assert account_id == "acc_new_123"
        
        # Verify account stored
        accounts = vault.get("mono", "accounts")
        assert "acc_new_123" in accounts["ids"]
    
    def test_mono_exchange_code_failure(self, tmp_path):
        """Mono exchange_code raises on failure."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        vault.store("mono", "secret_key", {"key": "test_sk_123"})
        
        # Mock failed response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status = 401
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            with pytest.raises(Exception, match="Code exchange failed"):
                connector.exchange_code("invalid_code")
    
    def test_mono_get_account(self, tmp_path):
        """Mono get_account returns BankAccount."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        vault.store("mono", "secret_key", {"key": "test_sk_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "account": {
                "institution": {"name": "GTBank"},
                "account_number": "0123456789",
                "type": "checking",
                "balance": 150000.50  # Naira
            }
        }
        
        with patch.object(connector.http, "get", return_value=mock_response):
            account = connector.get_account("acc_123")
        
        assert isinstance(account, BankAccount)
        assert account.provider == "mono"
        assert account.institution == "GTBank"
        assert account.mask == "6789"
        assert account.balance_minor == 15000050  # Kobo
        assert account.currency == "NGN"
    
    def test_mono_get_transactions(self, tmp_path):
        """Mono get_transactions returns list of transactions."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        vault.store("mono", "secret_key", {"key": "test_sk_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "transactions": [
                {
                    "id": "tx_1",
                    "amount": 5000.0,
                    "narration": "Transfer to John",
                    "date": "2026-09-28",
                    "type": "debit"
                },
                {
                    "id": "tx_2",
                    "amount": 150000.0,
                    "narration": "Salary",
                    "date": "2026-09-25",
                    "type": "credit"
                }
            ]
        }
        
        with patch.object(connector.http, "get", return_value=mock_response):
            transactions = connector.get_transactions("acc_123", days=30)
        
        assert len(transactions) == 2
        assert all(isinstance(tx, Transaction) for tx in transactions)
        
        tx1 = transactions[0]
        assert tx1.transaction_id == "tx_1"
        assert tx1.amount_minor == 500000  # Kobo
        assert tx1.currency == "NGN"
        assert "Transfer" in tx1.description
    
    def test_mono_verify_webhook_valid(self, tmp_path):
        """Mono verify_webhook returns True for valid signature."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        # Store webhook secret
        vault.store("mono", "webhook_secret", {"secret": "whsec_test_123"})
        
        # Create payload and signature
        payload = b'{"event": "account.updated"}'
        
        import hashlib
        import hmac
        expected_sig = hmac.new(
            b"whsec_test_123",
            payload,
            hashlib.sha512
        ).hexdigest()
        
        # Verify
        valid = connector.verify_webhook(payload, expected_sig)
        assert valid is True
    
    def test_mono_verify_webhook_invalid(self, tmp_path):
        """Mono verify_webhook returns False for invalid signature."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = MonoConnector(vault=vault)
        
        vault.store("mono", "webhook_secret", {"secret": "whsec_test_123"})
        
        payload = b'{"event": "account.updated"}'
        invalid_sig = "invalid_signature_12345"
        
        valid = connector.verify_webhook(payload, invalid_sig)
        assert valid is False


class TestPlaidConnector:
    """Test Plaid connector with mocked HTTP."""
    
    def test_plaid_status_no_keys(self, tmp_path):
        """Plaid status returns disconnected when no keys."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PlaidConnector(vault=vault)
        
        status = connector.status()
        assert not status.connected
        assert "No Plaid API keys" in status.error
    
    def test_plaid_exchange_public_token(self, tmp_path):
        """Plaid exchange_public_token returns access token."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PlaidConnector(vault=vault)
        
        vault.store("plaid", "api_keys", {
            "client_id": "client_123",
            "secret": "secret_456"
        })
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "access_token": "access_test_789",
            "item_id": "item_abc"
        }
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            access_token = connector.exchange_public_token("public_test_123")
        
        assert access_token == "access_test_789"
        
        # Verify stored
        accounts = vault.get("plaid", "accounts")
        assert "access_test_789" in accounts["tokens"]


class TestFinanceFacade:
    """Test unified Finance facade."""
    
    def test_finance_link_mono(self, tmp_path):
        """Finance.link() returns Mono connect URL."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        vault.store("mono", "public_key", {"key": "pk_test_123"})
        
        finance = Finance(vault=vault)
        result = finance.link(provider="mono")
        
        assert "connect_url" in result
        assert result["provider"] == "mono"
        assert "pk_test_123" in result["connect_url"]
    
    def test_finance_accounts_empty(self, tmp_path):
        """Finance.accounts() returns empty list when no accounts."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        finance = Finance(vault=vault)
        
        accounts = finance.accounts()
        assert accounts == []
    
    def test_finance_accounts_with_mono(self, tmp_path):
        """Finance.accounts() returns Mono accounts."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        vault.store("mono", "secret_key", {"key": "sk_test_123"})
        vault.store("mono", "accounts", {"ids": ["acc_123"]})
        
        finance = Finance(vault=vault)
        
        # Mock get_account
        with patch.object(finance.mono, "get_account") as mock_get:
            mock_get.return_value = BankAccount(
                account_id="acc_123",
                provider="mono",
                institution="GTBank",
                mask="6789",
                account_type="checking",
                balance_minor=15000000,
                currency="NGN"
            )
            
            accounts = finance.accounts()
        
        assert len(accounts) == 1
        assert accounts[0]["account_id"] == "acc_123"
        assert accounts[0]["institution"] == "GTBank"
    
    def test_finance_transactions(self, tmp_path):
        """Finance.transactions() returns transaction list."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        finance = Finance(vault=vault)
        
        # Mock get_transactions
        with patch.object(finance.mono, "get_transactions") as mock_get:
            mock_get.return_value = [
                Transaction(
                    transaction_id="tx_1",
                    account_id="acc_123",
                    amount_minor=500000,
                    currency="NGN",
                    description="Transfer",
                    date="2026-09-28"
                )
            ]
            
            txs = finance.transactions("acc_123", days=30)
        
        assert len(txs) == 1
        assert txs[0]["transaction_id"] == "tx_1"
        assert txs[0]["amount_minor"] == 500000


class TestBankAccount:
    """Test BankAccount dataclass."""
    
    def test_bank_account_to_dict(self):
        """BankAccount.to_dict() returns complete dict."""
        account = BankAccount(
            account_id="acc_123",
            provider="mono",
            institution="GTBank",
            mask="6789",
            account_type="checking",
            balance_minor=15000000,
            currency="NGN",
            linked_at=1696000000.0
        )
        
        data = account.to_dict()
        assert data["account_id"] == "acc_123"
        assert data["provider"] == "mono"
        assert data["institution"] == "GTBank"
        assert data["mask"] == "6789"
        assert data["balance_minor"] == 15000000
        assert data["currency"] == "NGN"


class TestTransaction:
    """Test Transaction dataclass."""
    
    def test_transaction_to_dict(self):
        """Transaction.to_dict() returns complete dict."""
        tx = Transaction(
            transaction_id="tx_123",
            account_id="acc_456",
            amount_minor=-500000,  # Debit
            currency="NGN",
            description="Transfer to John",
            date="2026-09-28",
            category="debit",
            merchant="John Doe",
            pending=False
        )
        
        data = tx.to_dict()
        assert data["transaction_id"] == "tx_123"
        assert data["amount_minor"] == -500000
        assert data["description"] == "Transfer to John"
        assert data["pending"] is False
