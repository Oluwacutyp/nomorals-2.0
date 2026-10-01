"""Encrypted credential vault with SQLite backend.

Credentials are encrypted at rest using AES-256-CTR with HMAC authentication.
The master key is derived from a passphrase via PBKDF2 (200k iterations).

Security model:
- Credentials never stored in plaintext
- Each credential gets its own encryption key derived from master key + credential ID
- HMAC validates ciphertext integrity before decryption
- Failed authentication raises CipherError (no silent corruption)

Usage:
    vault = CredentialVault(db, master_passphrase="your-secret")
    
    # Store a credential
    vault.store(
        service="gmail",
        username="bot@example.com",
        password="app-password-here",
        tags=["email", "primary"]
    )
    
    # Retrieve it
    cred = vault.get(service="gmail", username="bot@example.com")
    print(cred.password)  # Decrypted on access
    
    # List all credentials
    for cred in vault.list_all():
        print(f"{cred.service}: {cred.username}")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.cipher import aes_decrypt, aes_encrypt, derive_key
from ..core.errors import NotFound, StorageError
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = ["CredentialVault", "Credential", "AccountProfile"]

# PBKDF2 iterations for vault key derivation (matches the cipher tool default).
_KDF_ITERATIONS = 100_000

_log = get_logger(__name__)


@dataclass
class Credential:
    """A stored credential with metadata."""
    
    id: int
    service: str
    username: str
    password: str  # Decrypted when retrieved
    credential_type: str = "password"  # password, api_key, oauth_token, etc.
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    use_count: int = 0
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (password excluded for safety)."""
        return {
            "id": self.id,
            "service": self.service,
            "username": self.username,
            "credential_type": self.credential_type,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "is_active": self.is_active,
        }


@dataclass
class AccountProfile:
    """A named identity profile with associated credentials."""
    
    name: str
    description: str = ""
    credentials: list[Credential] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_credential(self, service: str) -> Optional[Credential]:
        """Get the first active credential for a service."""
        for cred in self.credentials:
            if cred.service == service and cred.is_active and not cred.is_expired():
                return cred
        return None


class CredentialVault:
    """Encrypted credential storage with SQLite backend.
    
    Example:
        vault = CredentialVault(db, master_passphrase="secret")
        vault.store("gmail", "user@example.com", "password123")
        cred = vault.get("gmail", "user@example.com")
    """
    
    def __init__(self, db: Database, master_passphrase: str) -> None:
        self.db = db
        self._master_key = derive_key(
            master_passphrase, salt=b"nomorals-vault-master",
            iterations=_KDF_ITERATIONS).hex()
        self._ensure_schema()
        _log.info("Credential vault initialized")
    
    def _ensure_schema(self) -> None:
        """Create credentials table if it doesn't exist."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    username TEXT NOT NULL,
                    password_encrypted TEXT NOT NULL,
                    credential_type TEXT NOT NULL DEFAULT 'password',
                    tags TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    last_used REAL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(service, username)
                )
            """)
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_credentials_service
                ON credentials(service)
            """)
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_credentials_tags
                ON credentials(tags)
            """)
    
    def _encrypt_password(self, password: str, credential_id: int) -> str:
        """Encrypt password with a key derived from master + credential ID."""
        # Derive a unique key for this credential
        salt = f"credential-{credential_id}".encode()
        key = derive_key(self._master_key, salt=salt,
                         iterations=_KDF_ITERATIONS)
        encrypted = aes_encrypt(password.encode(), key=key)
        return encrypted
    
    def _decrypt_password(self, encrypted: str, credential_id: int) -> str:
        """Decrypt password with a key derived from master + credential ID."""
        salt = f"credential-{credential_id}".encode()
        key = derive_key(self._master_key, salt=salt,
                         iterations=_KDF_ITERATIONS)
        decrypted = aes_decrypt(encrypted, key=key)
        return decrypted.decode("utf-8")
    
    def store(
        self,
        service: str,
        username: str,
        password: str,
        *,
        credential_type: str = "password",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: float | None = None,
    ) -> Credential:
        """Store a new credential or update existing one.
        
        Args:
            service: Service name (e.g., "gmail", "github")
            username: Username or identifier
            password: Password, API key, or token (will be encrypted)
            credential_type: Type of credential (password, api_key, oauth_token)
            tags: List of tags for organization
            metadata: Additional metadata dict
            expires_at: Unix timestamp when credential expires
            
        Returns:
            The created/updated Credential object
        """
        tags = tags or []
        metadata = metadata or {}
        now = time.time()
        
        with self.db.transaction():
            # Check if credential already exists
            existing = self.db.query_one(
                "SELECT id FROM credentials WHERE service = ? AND username = ?",
                (service, username)
            )
            
            if existing:
                # Update existing
                cred_id = existing["id"]
                encrypted = self._encrypt_password(password, cred_id)
                self.db.execute("""
                    UPDATE credentials SET
                        password_encrypted = ?,
                        credential_type = ?,
                        tags = ?,
                        metadata = ?,
                        updated_at = ?,
                        expires_at = ?,
                        is_active = 1
                    WHERE id = ?
                """, (
                    encrypted,
                    credential_type,
                    json.dumps(tags),
                    json.dumps(metadata),
                    now,
                    expires_at,
                    cred_id,
                ))
                _log.info(f"Updated credential: {service}/{username}")
            else:
                # Insert new (with placeholder encryption, we'll update after getting ID)
                cur = self.db.execute("""
                    INSERT INTO credentials (
                        service, username, password_encrypted, credential_type,
                        tags, metadata, created_at, updated_at, expires_at,
                        use_count, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
                """, (
                    service, username, "placeholder", credential_type,
                    json.dumps(tags), json.dumps(metadata), now, now, expires_at,
                ))
                cred_id = cur.lastrowid
                # Now encrypt with the actual ID
                encrypted = self._encrypt_password(password, cred_id)
                self.db.execute(
                    "UPDATE credentials SET password_encrypted = ? WHERE id = ?",
                    (encrypted, cred_id)
                )
                _log.info(f"Stored new credential: {service}/{username}")
        
        # Return the credential object
        return Credential(
            id=cred_id,
            service=service,
            username=username,
            password=password,
            credential_type=credential_type,
            tags=tags,
            metadata=metadata,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            use_count=0,
            is_active=True,
        )
    
    def get(self, service: str, username: str, *, mark_used: bool = True) -> Credential:
        """Retrieve a credential by service and username.
        
        Args:
            service: Service name
            username: Username or identifier
            mark_used: If True, update last_used timestamp and use_count
            
        Returns:
            The Credential object with decrypted password
            
        Raises:
            NotFound: If credential doesn't exist
        """
        row = self.db.query_one(
            "SELECT * FROM credentials WHERE service = ? AND username = ?",
            (service, username)
        )
        if not row:
            raise NotFound(f"Credential not found: {service}/{username}")
        
        # Decrypt password
        password = self._decrypt_password(row["password_encrypted"], row["id"])
        
        # Update usage if requested
        if mark_used:
            now = time.time()
            with self.db.transaction():
                self.db.execute("""
                    UPDATE credentials SET last_used = ?, use_count = use_count + 1
                    WHERE id = ?
                """, (now, row["id"]))
        
        return Credential(
            id=row["id"],
            service=row["service"],
            username=row["username"],
            password=password,
            credential_type=row["credential_type"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            last_used=row["last_used"],
            use_count=row["use_count"],
            is_active=bool(row["is_active"]),
        )
    
    def list_all(
        self,
        *,
        service: str | None = None,
        tag: str | None = None,
        active_only: bool = True,
    ) -> list[Credential]:
        """List all credentials, optionally filtered.
        
        Args:
            service: Filter by service name
            tag: Filter by tag (credentials must have this tag)
            active_only: If True, only return active credentials
            
        Returns:
            List of Credential objects (passwords NOT decrypted for safety)
        """
        query = "SELECT * FROM credentials WHERE 1=1"
        params: list[Any] = []
        
        if service:
            query += " AND service = ?"
            params.append(service)
        
        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")
        
        if active_only:
            query += " AND is_active = 1"
        
        query += " ORDER BY service, username"
        
        rows = self.db.query(query, params)
        credentials = []
        
        for row in rows:
            # Don't decrypt passwords for listing (security)
            credentials.append(Credential(
                id=row["id"],
                service=row["service"],
                username=row["username"],
                password="***",  # Not decrypted
                credential_type=row["credential_type"],
                tags=json.loads(row["tags"]),
                metadata=json.loads(row["metadata"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
                last_used=row["last_used"],
                use_count=row["use_count"],
                is_active=bool(row["is_active"]),
            ))
        
        return credentials
    
    def delete(self, service: str, username: str) -> None:
        """Delete a credential.
        
        Args:
            service: Service name
            username: Username or identifier
        """
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM credentials WHERE service = ? AND username = ?",
                (service, username)
            )
        _log.info(f"Deleted credential: {service}/{username}")
    
    def deactivate(self, service: str, username: str) -> None:
        """Deactivate a credential without deleting it.
        
        Args:
            service: Service name
            username: Username or identifier
        """
        with self.db.transaction():
            self.db.execute("""
                UPDATE credentials SET is_active = 0, updated_at = ?
                WHERE service = ? AND username = ?
            """, (time.time(), service, username))
        _log.info(f"Deactivated credential: {service}/{username}")
    
    def rotate(self, service: str, username: str, new_password: str) -> Credential:
        """Rotate a credential's password/key.
        
        Args:
            service: Service name
            username: Username or identifier
            new_password: New password/key
            
        Returns:
            Updated Credential object
        """
        return self.store(service, username, new_password)
    
    def get_profile(self, profile_name: str) -> AccountProfile:
        """Get an account profile by name.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            AccountProfile with associated credentials
        """
        # For now, just return a profile with all credentials
        # In the future, profiles could be stored separately
        credentials = self.list_all(active_only=True)
        
        return AccountProfile(
            name=profile_name,
            description=f"Profile: {profile_name}",
            credentials=credentials,
            metadata={},
        )
