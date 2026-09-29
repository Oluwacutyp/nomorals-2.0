"""Tests for connector framework — vault, base, patterns."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from nomorals.connectors.base import BaseConnector, ConnectorStatus
from nomorals.connectors.vault import CredentialVault, VaultError
from nomorals.connectors.patterns import ConnectionPattern, PatternRegistry, patterns


class TestCredentialVault:
    """Test encrypted credential vault."""
    
    def test_vault_encrypt_decrypt_roundtrip(self, tmp_path):
        """Vault encrypts and decrypts correctly."""
        db_path = tmp_path / "test_vault.db"
        vault = CredentialVault(db_path=db_path)
        
        # Store credential
        secret = {"api_key": "sk_test_1234567890abcdef", "account_id": "acc_123"}
        vault.store("test_connector", "api_key", secret)
        
        # Retrieve and verify
        retrieved = vault.get("test_connector", "api_key")
        assert retrieved == secret
        assert retrieved["api_key"] == "sk_test_1234567890abcdef"
    
    def test_vault_never_leaks_values_in_list(self, tmp_path):
        """list_labels() returns labels only, never values."""
        db_path = tmp_path / "test_vault.db"
        vault = CredentialVault(db_path=db_path)
        
        # Store multiple credentials
        vault.store("mono", "secret_key", {"key": "sk_live_abc123"})
        vault.store("mono", "public_key", {"key": "pk_live_xyz789"})
        vault.store("plaid", "api_keys", {"client_id": "client_123", "secret": "secret_456"})
        
        # List labels
        mono_labels = vault.list_labels("mono")
        plaid_labels = vault.list_labels("plaid")
        
        # Verify labels only, no values
        assert mono_labels == ["public_key", "secret_key"]
        assert plaid_labels == ["api_keys"]
        
        # Ensure no secret values in labels
        all_labels = mono_labels + plaid_labels
        for label in all_labels:
            assert "sk_live" not in label
            assert "pk_live" not in label
            assert "client_123" not in label
            assert "secret_456" not in label
    
    def test_vault_delete_credential(self, tmp_path):
        """Vault deletes credentials correctly."""
        db_path = tmp_path / "test_vault.db"
        vault = CredentialVault(db_path=db_path)
        
        # Store and verify
        vault.store("test", "temp", {"data": "temporary"})
        assert vault.get("test", "temp") is not None
        
        # Delete
        deleted = vault.delete("test", "temp")
        assert deleted is True
        
        # Verify deleted
        assert vault.get("test", "temp") is None
    
    def test_vault_clear_connector(self, tmp_path):
        """Vault clears all credentials for a connector."""
        db_path = tmp_path / "test_vault.db"
        vault = CredentialVault(db_path=db_path)
        
        # Store multiple credentials
        vault.store("mono", "key1", {"data": "1"})
        vault.store("mono", "key2", {"data": "2"})
        vault.store("plaid", "key1", {"data": "3"})
        
        # Clear mono only
        count = vault.clear("mono")
        assert count == 2
        
        # Verify mono cleared, plaid intact
        assert vault.get("mono", "key1") is None
        assert vault.get("mono", "key2") is None
        assert vault.get("plaid", "key1") is not None
    
    def test_vault_exception_never_leaks_secrets(self, tmp_path):
        """Vault exceptions never contain secret values."""
        db_path = tmp_path / "test_vault.db"
        vault = CredentialVault(db_path=db_path)
        
        # Store a secret
        secret_value = "super_secret_api_key_12345"
        vault.store("test", "secret", {"key": secret_value})
        
        # Try to trigger an exception (e.g., invalid decryption)
        # Corrupt the database
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE vault SET encrypted_data = 'corrupted'")
        conn.commit()
        conn.close()
        
        # Try to retrieve - should raise VaultError
        with pytest.raises(VaultError) as exc_info:
            vault.get("test", "secret")
        
        # Verify secret not in exception
        assert secret_value not in str(exc_info.value)
        assert "super_secret" not in str(exc_info.value)


class TestBaseConnector:
    """Test base connector interface."""
    
    def test_connector_status_never_cached(self, tmp_path):
        """status() returns fresh results every call."""
        
        class MockConnector(BaseConnector):
            name = "mock"
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
            
            def status(self):
                self.call_count += 1
                return ConnectorStatus(
                    connected=True,
                    account=f"call_{self.call_count}"
                )
            
            def connect_url(self):
                return ""
            
            def disconnect(self):
                return {"disconnected": True}
        
        connector = MockConnector()
        
        # Call status twice
        status1 = connector.status()
        status2 = connector.status()
        
        # Verify fresh results
        assert status1.account == "call_1"
        assert status2.account == "call_2"
        assert connector.call_count == 2
    
    def test_connector_capabilities_scope_honesty(self, tmp_path):
        """capabilities() only lists implemented features."""
        
        class MinimalConnector(BaseConnector):
            name = "minimal"
            
            def status(self):
                return ConnectorStatus(connected=True)
            
            def connect_url(self):
                return ""
            
            def disconnect(self):
                return {"disconnected": True}
            
            def capabilities(self):
                # Only list what's actually implemented
                return ["read_data"]
            
            def read_data(self):
                return {"data": "test"}
        
        connector = MinimalConnector()
        caps = connector.capabilities()
        
        # Verify only implemented capability listed
        assert caps == ["read_data"]
        assert "write_data" not in caps
        assert "delete_data" not in caps
    
    def test_connector_vault_integration(self, tmp_path):
        """Connector integrates with vault for credential storage."""
        db_path = tmp_path / "test_vault.db"
        vault = CredentialVault(db_path=db_path)
        
        class VaultConnector(BaseConnector):
            name = "vault_test"
            
            def status(self):
                creds = self._get_credential("api_key")
                if creds and creds.get("key"):
                    return ConnectorStatus(connected=True)
                return ConnectorStatus(connected=False, error="No API key")
            
            def connect_url(self):
                return ""
            
            def disconnect(self):
                self._delete_credential("api_key")
                return {"disconnected": True}
        
        connector = VaultConnector(vault=vault)
        
        # Initially disconnected
        status = connector.status()
        assert not status.connected
        
        # Store credential
        connector._store_credential("api_key", {"key": "test_key_123"})
        
        # Now connected
        status = connector.status()
        assert status.connected
        
        # Disconnect
        connector.disconnect()
        status = connector.status()
        assert not status.connected


class TestPatternRegistry:
    """Test connection pattern registry."""
    
    def test_pattern_registry_lists_all_10_patterns(self):
        """Registry lists all 10 connection patterns."""
        registry = PatternRegistry()
        pattern_list = registry.list()
        
        # Verify all 10 patterns
        assert len(pattern_list) == 10
        
        # Verify specific patterns
        pattern_names = [p.pattern for p in pattern_list]
        assert ConnectionPattern.OAUTH_ACCOUNTS_CENTER in pattern_names
        assert ConnectionPattern.OAUTH_PROVIDER_HOSTED in pattern_names
        assert ConnectionPattern.API_KEY_VAULT in pattern_names
        assert ConnectionPattern.BROWSER_FALLBACK in pattern_names
    
    def test_pattern_descriptions_not_empty(self):
        """All patterns have descriptions."""
        registry = PatternRegistry()
        
        for pattern in ConnectionPattern:
            desc = registry.describe(pattern)
            assert desc, f"Pattern {pattern} has no description"
            assert len(desc) > 10, f"Pattern {pattern} description too short"
    
    def test_pattern_get_returns_config(self):
        """get() returns pattern config."""
        registry = PatternRegistry()
        
        config = registry.get(ConnectionPattern.API_KEY_VAULT)
        assert config is not None
        assert config.pattern == ConnectionPattern.API_KEY_VAULT
        assert "API key" in config.description


class TestConnectorStatus:
    """Test ConnectorStatus dataclass."""
    
    def test_status_to_dict(self):
        """status.to_dict() returns complete dict."""
        status = ConnectorStatus(
            connected=True,
            account="test_account",
            scopes=["read", "write"],
            error=""
        )
        
        data = status.to_dict()
        assert data["connected"] is True
        assert data["account"] == "test_account"
        assert data["scopes"] == ["read", "write"]
        assert data["error"] == ""
        assert "last_check" in data
    
    def test_status_auto_timestamp(self):
        """status auto-generates timestamp."""
        before = time.time()
        status = ConnectorStatus(connected=True)
        after = time.time()
        
        # Verify timestamp in ISO format
        assert status.last_check
        assert "T" in status.last_check  # ISO format has T separator
        
        # Verify timestamp is recent
        from datetime import datetime
        ts = datetime.fromisoformat(status.last_check.replace("Z", "+00:00"))
        ts_epoch = ts.timestamp()
        assert before <= ts_epoch <= after + 1
