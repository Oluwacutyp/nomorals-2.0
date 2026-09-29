"""Encrypted credential vault — secrets at rest, never in logs/output."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = ["CredentialVault", "VaultError"]

_log = get_logger(__name__)


class VaultError(Exception):
    """Vault operation failed."""
    pass


class CredentialVault:
    """Encrypted credential storage.
    
    Security rules:
    - Fernet encryption (symmetric, AES-128-CBC + HMAC)
    - Key from NM_VAULT_KEY env var or auto-generated on first run
    - Secrets never in logs, output, memory files, or exceptions
    - Card PANs/CVVs masked in all storage except checkout handoff
    
    API:
    - store(connector, label, secret_dict)
    - get(connector, label) → dict | None
    - delete(connector, label) → bool
    - list_labels(connector) → list[str] (labels only, never values)
    """
    
    def __init__(self, db_path: str | Path | None = None, key: str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path.home() / ".nomorals" / "vault.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get or generate encryption key
        self._key = key or os.environ.get("NM_VAULT_KEY") or self._load_or_generate_key()
        self._fernet = self._init_fernet(self._key)
        
        # Initialize database
        self._init_db()
    
    def _load_or_generate_key(self) -> str:
        """Load key from file or generate new one."""
        key_file = self.db_path.parent / ".vault_key"
        if key_file.exists():
            return key_file.read_text().strip()
        
        # Generate new key
        import secrets
        new_key = secrets.token_urlsafe(32)
        key_file.write_text(new_key)
        key_file.chmod(0o600)  # Owner read/write only
        _log.info("Generated new vault key")
        return new_key
    
    def _init_fernet(self, key: str) -> Any:
        """Initialize Fernet cipher. Falls back to base64 if cryptography not available."""
        try:
            from cryptography.fernet import Fernet
            # Fernet needs 32 url-safe base64-encoded bytes
            fernet_key = hashlib.sha256(key.encode()).digest()
            import base64
            fernet_key_b64 = base64.urlsafe_b64encode(fernet_key)
            return Fernet(fernet_key_b64)
        except ImportError:
            _log.warning("cryptography not installed, vault using base64 (NOT SECURE)")
            return None
    
    def _init_db(self) -> None:
        """Create vault table if not exists."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vault (
                    connector TEXT NOT NULL,
                    label TEXT NOT NULL,
                    encrypted_data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (connector, label)
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def _encrypt(self, data: str) -> str:
        """Encrypt string data."""
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        else:
            # Fallback: base64 (NOT SECURE, only for testing)
            import base64
            return base64.b64encode(data.encode()).decode()
    
    def _decrypt(self, encrypted: str) -> str:
        """Decrypt string data."""
        if self._fernet:
            return self._fernet.decrypt(encrypted.encode()).decode()
        else:
            # Fallback: base64
            import base64
            return base64.b64decode(encrypted.encode()).decode()
    
    def store(self, connector: str, label: str, secret_dict: dict[str, Any]) -> None:
        """Store encrypted credential.
        
        Args:
            connector: Connector name (e.g., "mono", "privacy_cards")
            label: Credential label (e.g., "api_key", "oauth_token")
            secret_dict: Dict of secrets to encrypt
        
        Raises:
            VaultError: If storage fails
        """
        try:
            import time
            json_data = json.dumps(secret_dict, separators=(",", ":"))
            encrypted = self._encrypt(json_data)
            
            conn = sqlite3.connect(str(self.db_path))
            try:
                now = time.time()
                conn.execute("""
                    INSERT OR REPLACE INTO vault 
                    (connector, label, encrypted_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (connector, label, encrypted, now, now))
                conn.commit()
            finally:
                conn.close()
            
            _log.debug("Stored credential: %s/%s", connector, label)
        
        except Exception as e:
            # Never leak secret values in exceptions
            raise VaultError(f"Failed to store credential: {type(e).__name__}") from e
    
    def get(self, connector: str, label: str) -> dict[str, Any] | None:
        """Retrieve and decrypt credential.
        
        Returns:
            Decrypted dict, or None if not found
        
        Raises:
            VaultError: If decryption fails
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(
                    "SELECT encrypted_data FROM vault WHERE connector = ? AND label = ?",
                    (connector, label)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                
                encrypted = row[0]
                decrypted = self._decrypt(encrypted)
                return json.loads(decrypted)
            finally:
                conn.close()
        
        except Exception as e:
            # Never leak secret values in exceptions
            raise VaultError(f"Failed to retrieve credential: {type(e).__name__}") from e
    
    def delete(self, connector: str, label: str) -> bool:
        """Delete credential.
        
        Returns:
            True if deleted, False if not found
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(
                    "DELETE FROM vault WHERE connector = ? AND label = ?",
                    (connector, label)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
        
        except Exception as e:
            raise VaultError(f"Failed to delete credential: {type(e).__name__}") from e
    
    def list_labels(self, connector: str) -> list[str]:
        """List credential labels for a connector (never values).
        
        Returns:
            List of label strings
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(
                    "SELECT label FROM vault WHERE connector = ? ORDER BY label",
                    (connector,)
                )
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
        
        except Exception as e:
            raise VaultError(f"Failed to list credentials: {type(e).__name__}") from e
    
    def clear(self, connector: str | None = None) -> int:
        """Clear credentials. If connector specified, only that one.
        
        Returns:
            Number of credentials deleted
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                if connector:
                    cursor = conn.execute("DELETE FROM vault WHERE connector = ?", (connector,))
                else:
                    cursor = conn.execute("DELETE FROM vault")
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
        
        except Exception as e:
            raise VaultError(f"Failed to clear credentials: {type(e).__name__}") from e
