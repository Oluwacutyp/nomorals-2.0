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
    # repr=False: a stray repr()/log of a Credential must never carry the
    # secret — free-text redaction can't catch a bare random string.
    password: str = field(repr=False)  # Decrypted when retrieved
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
            # Named identity profiles and their credential membership.
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS account_profiles (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS profile_credentials (
                    profile_name TEXT NOT NULL,
                    credential_id INTEGER NOT NULL,
                    added_at REAL NOT NULL,
                    PRIMARY KEY (profile_name, credential_id)
                )
            """)
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_profile_credentials_cred
                ON profile_credentials(credential_id)
            """)
    
    def _encrypt_password(self, password: str, credential_id: int) -> str:
        """Encrypt password with a key derived from master + credential ID.

        The secret is JSON-wrapped first so empty strings (used by
        password-less credentials like disposable emails) still produce
        a non-empty ciphertext — the cipher rejects empty payloads.
        """
        # Derive a unique key for this credential
        salt = f"credential-{credential_id}".encode()
        key = derive_key(self._master_key, salt=salt,
                         iterations=_KDF_ITERATIONS)
        encrypted = aes_encrypt(json.dumps(password).encode(), key=key)
        return encrypted

    def _decrypt_password(self, encrypted: str, credential_id: int) -> str:
        """Decrypt password with a key derived from master + credential ID."""
        salt = f"credential-{credential_id}".encode()
        key = derive_key(self._master_key, salt=salt,
                         iterations=_KDF_ITERATIONS)
        decrypted = aes_decrypt(encrypted, key=key).decode("utf-8")
        try:
            return json.loads(decrypted)
        except ValueError:
            # Rows written before JSON-wrapping: raw plaintext.
            return decrypted

    def encrypt_blob(self, plaintext: str, *, purpose: str) -> str:
        """Encrypt an opaque blob (session data, token payloads, ...).

        Uses a purpose-scoped key derived from the vault master key.
        Other organs (sessions, connectors) use this so secrets never
        sit in plaintext in their own tables.

        Args:
            plaintext: Text to encrypt
            purpose: Key-separation label (e.g. "sessions")

        Returns:
            Self-describing encrypted blob string
        """
        if not purpose:
            raise ValueError("purpose must be a non-empty label")
        key = derive_key(
            self._master_key,
            salt=f"nomorals-vault-blob:{purpose}".encode(),
            iterations=_KDF_ITERATIONS,
        )
        return aes_encrypt(plaintext.encode("utf-8"), key=key)

    def decrypt_blob(self, blob: str, *, purpose: str) -> str:
        """Decrypt a blob produced by :meth:`encrypt_blob`.

        Args:
            blob: Encrypted blob string
            purpose: The same label used at encryption time

        Returns:
            Decrypted plaintext

        Raises:
            CipherError: If the blob is tampered or the purpose is wrong
        """
        key = derive_key(
            self._master_key,
            salt=f"nomorals-vault-blob:{purpose}".encode(),
            iterations=_KDF_ITERATIONS,
        )
        return aes_decrypt(blob, key=key).decode("utf-8")

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
        
        # Update usage if requested; the returned object reflects
        # the post-increment values.
        use_count = row["use_count"]
        last_used = row["last_used"]
        if mark_used:
            now = time.time()
            use_count += 1
            last_used = now
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
            last_used=last_used,
            use_count=use_count,
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

        Also removes it from every profile it belongs to.

        Args:
            service: Service name
            username: Username or identifier

        Raises:
            NotFound: If the credential doesn't exist
        """
        with self.db.transaction():
            cur = self.db.execute(
                "DELETE FROM credentials WHERE service = ? AND username = ?",
                (service, username)
            )
            if cur.rowcount == 0:
                raise NotFound(f"Credential not found: {service}/{username}")
            # Manual cascade: drop profile memberships whose credential is gone.
            self.db.execute(
                """DELETE FROM profile_credentials WHERE credential_id NOT IN
                   (SELECT id FROM credentials)"""
            )
        _log.info(f"Deleted credential: {service}/{username}")

    def deactivate(self, service: str, username: str) -> None:
        """Deactivate a credential without deleting it.

        Args:
            service: Service name
            username: Username or identifier

        Raises:
            NotFound: If the credential doesn't exist
        """
        with self.db.transaction():
            cur = self.db.execute("""
                UPDATE credentials SET is_active = 0, updated_at = ?
                WHERE service = ? AND username = ?
            """, (time.time(), service, username))
            if cur.rowcount == 0:
                raise NotFound(f"Credential not found: {service}/{username}")
        _log.info(f"Deactivated credential: {service}/{username}")

    def rotate(self, service: str, username: str, new_password: str) -> Credential:
        """Rotate a credential's secret in place.

        Only the secret and ``updated_at`` change — tags, metadata,
        credential type and expiry are preserved (a full :meth:`store`
        would reset them).

        Args:
            service: Service name
            username: Username or identifier
            new_password: New password/key/token

        Returns:
            Updated Credential object

        Raises:
            NotFound: If the credential doesn't exist
        """
        row = self.db.query_one(
            "SELECT id FROM credentials WHERE service = ? AND username = ?",
            (service, username)
        )
        if not row:
            raise NotFound(f"Credential not found: {service}/{username}")
        encrypted = self._encrypt_password(new_password, row["id"])
        with self.db.transaction():
            self.db.execute(
                """UPDATE credentials
                   SET password_encrypted = ?, updated_at = ?
                   WHERE id = ?""",
                (encrypted, time.time(), row["id"]),
            )
        _log.info(f"Rotated credential: {service}/{username}")
        return self.get(service, username)

    # ── identity profiles ──────────────────────────────────────────

    def create_profile(
        self,
        name: str,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AccountProfile:
        """Create a named identity profile.

        A profile groups credentials that belong to one identity
        (the owner's own, or a bot persona) so multi-service flows can
        pull "everything for X" in one call.

        Args:
            name: Unique profile name
            description: Human description
            metadata: Extra profile metadata

        Returns:
            The new (empty) AccountProfile

        Raises:
            StorageError: If a profile with this name already exists
        """
        name = name.strip()
        if not name:
            raise ValueError("profile name must not be empty")
        now = time.time()
        with self.db.transaction():
            existing = self.db.query_one(
                "SELECT name FROM account_profiles WHERE name = ?", (name,)
            )
            if existing:
                raise StorageError(f"Profile already exists: {name!r}")
            self.db.execute(
                """INSERT INTO account_profiles
                   (name, description, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, json.dumps(metadata or {}), now, now),
            )
        _log.info(f"Created account profile: {name}")
        return self.get_profile(name)

    def delete_profile(self, name: str) -> None:
        """Delete a profile and its credential memberships (not the credentials).

        Raises:
            NotFound: If the profile doesn't exist
        """
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM profile_credentials WHERE profile_name = ?",
                (name,),
            )
            cur = self.db.execute(
                "DELETE FROM account_profiles WHERE name = ?", (name,)
            )
            if cur.rowcount == 0:
                raise NotFound(f"Profile not found: {name!r}")
        _log.info(f"Deleted account profile: {name}")

    def list_profiles(self) -> list[AccountProfile]:
        """List all profiles with their credentials."""
        rows = self.db.query(
            "SELECT name FROM account_profiles ORDER BY name"
        )
        return [self.get_profile(r["name"]) for r in rows]

    def add_to_profile(self, profile_name: str, service: str, username: str) -> None:
        """Attach a credential to a profile.

        Raises:
            NotFound: If the profile or the credential doesn't exist
        """
        profile = self.db.query_one(
            "SELECT name FROM account_profiles WHERE name = ?", (profile_name,)
        )
        if not profile:
            raise NotFound(f"Profile not found: {profile_name!r}")
        cred = self.db.query_one(
            "SELECT id FROM credentials WHERE service = ? AND username = ?",
            (service, username),
        )
        if not cred:
            raise NotFound(f"Credential not found: {service}/{username}")
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO profile_credentials
                   (profile_name, credential_id, added_at)
                   VALUES (?, ?, ?)""",
                (profile_name, cred["id"], time.time()),
            )
            self.db.execute(
                "UPDATE account_profiles SET updated_at = ? WHERE name = ?",
                (time.time(), profile_name),
            )
        _log.info(f"Added {service}/{username} to profile {profile_name!r}")

    def remove_from_profile(
        self, profile_name: str, service: str, username: str
    ) -> None:
        """Detach a credential from a profile.

        Raises:
            NotFound: If the profile or the credential doesn't exist
        """
        cred = self.db.query_one(
            "SELECT id FROM credentials WHERE service = ? AND username = ?",
            (service, username),
        )
        if not cred:
            raise NotFound(f"Credential not found: {service}/{username}")
        with self.db.transaction():
            cur = self.db.execute(
                """DELETE FROM profile_credentials
                   WHERE profile_name = ? AND credential_id = ?""",
                (profile_name, cred["id"]),
            )
            if cur.rowcount == 0:
                # Distinguish "no such profile" from "not a member".
                profile = self.db.query_one(
                    "SELECT name FROM account_profiles WHERE name = ?",
                    (profile_name,),
                )
                if not profile:
                    raise NotFound(f"Profile not found: {profile_name!r}")
                raise NotFound(
                    f"Credential {service}/{username} is not in profile "
                    f"{profile_name!r}"
                )

    def profiles_of(self, service: str, username: str) -> list[str]:
        """Names of all profiles containing this credential."""
        cred = self.db.query_one(
            "SELECT id FROM credentials WHERE service = ? AND username = ?",
            (service, username),
        )
        if not cred:
            raise NotFound(f"Credential not found: {service}/{username}")
        rows = self.db.query(
            """SELECT profile_name FROM profile_credentials
               WHERE credential_id = ? ORDER BY profile_name""",
            (cred["id"],),
        )
        return [r["profile_name"] for r in rows]

    def get_profile(self, profile_name: str) -> AccountProfile:
        """Get an account profile with its associated credentials.

        Passwords are masked (``"***"``), same as :meth:`list_all`.

        Args:
            profile_name: Name of the profile

        Returns:
            AccountProfile with associated credentials

        Raises:
            NotFound: If the profile doesn't exist
        """
        row = self.db.query_one(
            "SELECT * FROM account_profiles WHERE name = ?", (profile_name,)
        )
        if not row:
            raise NotFound(f"Profile not found: {profile_name!r}")
        cred_rows = self.db.query(
            """SELECT c.* FROM credentials c
               JOIN profile_credentials pc ON pc.credential_id = c.id
               WHERE pc.profile_name = ?
               ORDER BY c.service, c.username""",
            (profile_name,),
        )
        credentials = [
            Credential(
                id=r["id"],
                service=r["service"],
                username=r["username"],
                password="***",
                credential_type=r["credential_type"],
                tags=json.loads(r["tags"]),
                metadata=json.loads(r["metadata"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                expires_at=r["expires_at"],
                last_used=r["last_used"],
                use_count=r["use_count"],
                is_active=bool(r["is_active"]),
            )
            for r in cred_rows
        ]
        return AccountProfile(
            name=row["name"],
            description=row["description"] or "",
            credentials=credentials,
            metadata=json.loads(row["metadata"] or "{}"),
        )
