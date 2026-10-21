"""Encrypted credential vault with SQLite backend.

Credentials are encrypted at rest using AES-256-CTR with HMAC authentication.
The master key is derived from a passphrase via PBKDF2 (100k iterations).

Multi-credential unlock (Pass 4): the vault's master key can additionally
be *wrapped* (AES-encrypted under a key derived from another owner
credential) for each of the owner's standing credentials — the legacy
passphrase, the owner seal passphrase (``nomorals.core.owner``), and the
power-mode key (``partner.owner_key``).  :meth:`CredentialVault.unlock`
tries every available credential: any one whose wrap opens (or, for a
pre-wrap legacy vault, the passphrase whose derived key authenticates
against stored rows) unlocks the vault, and a successful unlock wraps
the master key under every other available verified credential, so
existing vaults migrate forward without a passphrase ceremony.  A wrong
credential fails closed — unwrapping is HMAC-authenticated, exactly as
hard as a wrong passphrase always was.

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
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.cipher import CipherError, aes_decrypt, aes_encrypt, derive_key
from ..core.errors import NotFound, StorageError
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = [
    "CredentialVault", "Credential", "AccountProfile",
    "VaultLockedError", "open_vault", "VAULT_CREDENTIAL_TYPES",
]

# PBKDF2 iterations for vault key derivation (matches the cipher tool default).
_KDF_ITERATIONS = 100_000

_log = get_logger(__name__)

#: Credential types that may unwrap the vault master key.  The owner set
#: all three: the legacy vault passphrase, the owner seal passphrase, and
#: the power-mode key — any ONE of them opens the vault.
VAULT_CREDENTIAL_TYPES = ("passphrase", "owner_seal", "power_key")

#: Human label + where the credential lives, per type — used in locked
#: error reports so an operator knows exactly what was checked where.
_CREDENTIAL_SOURCES = {
    "passphrase": "passphrase (NM_VAULT_PASSPHRASE environment variable)",
    "owner_seal": "owner seal (passphrase baked via `nm owner seal` "
                  "into nomorals/core/owner.py)",
    "power_key": "power key (partner.owner_key setting, env "
                 "NM_PARTNER_OWNER_KEY)",
}

_WRAP_SALT_PREFIX = b"nomorals-vault-wrap"
_VERIFIER_PLAINTEXT = b"nomorals-vault-verifier-v1"
_VERIFIER_SALT = b"nomorals-vault-verify"


class VaultLockedError(StorageError):
    """No available owner credential could open the vault.

    ``checks`` carries one line per credential source describing what
    was looked for and what happened, so the message is actionable
    without reading code.  Credential VALUES never appear in it.
    """

    code = "storage.vault_locked"
    retryable = False

    def __init__(self, checks: list[str]) -> None:
        self.checks = list(checks)
        lines = ["the credential vault is locked — no owner credential "
                 "could open it. Checked:"]
        lines.extend(f"  - {line}" for line in self.checks)
        lines.append(
            "any ONE of these opens the vault: the owner seal passphrase, "
            "the power key (partner.owner_key / NM_PARTNER_OWNER_KEY), or "
            "the legacy vault passphrase (NM_VAULT_PASSPHRASE).")
        super().__init__("\n".join(lines))

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

    @property
    def tier(self) -> str:
        """Proxy trust tier for this credential: ``free`` | ``paid`` | ``direct``.

        Read from ``metadata["tier"]`` (set by the proxy pool for proxy
        credentials; ``proxypool.add_proxy(..., tier=...)``).  Unlabelled
        credentials default to ``direct`` — no proxy is involved — while
        an unlabelled *proxy* credential is ``free`` (untrusted until
        proven otherwise).  Invalid values fail closed to ``free``.
        """
        from ..core.proxy_tiers import normalize_tier

        raw = (self.metadata or {}).get("tier")
        if raw is None or str(raw).strip() == "":
            if self.credential_type == "proxy":
                return "free"
            return "direct"
        try:
            return normalize_tier(raw)
        except ValueError:
            return "free"
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (password excluded for safety)."""
        return {
            "id": self.id,
            "service": self.service,
            "username": self.username,
            "credential_type": self.credential_type,
            "tier": self.tier,
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
            # Multi-credential unlock: the master key, wrapped once per
            # accepted owner credential, plus a verifier blob that lets
            # an unlock attempt prove a candidate master key is right
            # without touching credential rows.
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS vault_key_wraps (
                    credential_type TEXT PRIMARY KEY,
                    salt TEXT NOT NULL,
                    wrapped_key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS vault_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
    
    # ── multi-credential unlock: key wraps, verifier, audit ────────
    #
    # The master key itself never changes when credentials are added:
    # each accepted owner credential gets one row in vault_key_wraps —
    # the master key AES-encrypted under a key derived from that
    # credential (per-wrap random salt).  Unwrapping is authenticated
    # (CipherError on a wrong credential), so acceptance is real
    # cryptography, never a bypass flag.

    def _audit_unlock(self, actor: str, via: str, *, ok: bool,
                      why: str = "") -> None:
        """Audit-log an unlock outcome by credential TYPE only.

        The credential value is never written anywhere — ``via`` is one
        of :data:`VAULT_CREDENTIAL_TYPES` (or ``vault.migrate``)."""
        detail: dict[str, Any] = {"via": via}
        if why:
            detail["why"] = why
        try:
            from ..core.ids import ulid_now

            self.db.execute(
                "INSERT INTO audit_log (id, ts, actor, action, capability,"
                " decision, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid_now(), time.time(), actor or "vault",
                 "vault.unlock" if ok else "vault.unlock_denied",
                 "credential-vault", "allow" if ok else "deny",
                 json.dumps(detail)),
            )
        except Exception:  # noqa: BLE001 - audit must never break an unlock
            _log.debug("vault unlock audit write failed", exc_info=True)
        if ok:
            _log.info("vault unlocked via %s", via)
        else:
            _log.info("vault unlock denied via %s (%s)", via, why)

    def _wraps(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT credential_type, salt, wrapped_key FROM vault_key_wraps")
        return {r["credential_type"]: r for r in rows}

    def wrapped_credential_types(self) -> list[str]:
        """Credential types that currently hold a master-key wrap."""
        return sorted(self._wraps().keys())

    def _has_credentials(self) -> bool:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM credentials")
        return bool(row and row["n"])

    @staticmethod
    def _wrap_kek(credential_type: str, secret: str, salt: bytes) -> bytes:
        return derive_key(
            secret,
            salt=_WRAP_SALT_PREFIX + b":" + credential_type.encode("utf-8")
                 + b":" + salt,
            iterations=_KDF_ITERATIONS)

    def _wrap_master(self, credential_type: str, secret: str,
                     master_hex: str) -> None:
        salt = secrets.token_bytes(16)
        kek = self._wrap_kek(credential_type, secret, salt)
        wrapped = aes_encrypt(master_hex.encode("utf-8"), key=kek)
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                """INSERT OR REPLACE INTO vault_key_wraps
                   (credential_type, salt, wrapped_key, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (credential_type, salt.hex(), wrapped, now, now))

    def _unwrap_master(self, credential_type: str,
                       secret: str) -> str | None:
        """The master key unwrapped by ``secret``, or None when the
        credential is wrong (authentication failure) or no wrap exists."""
        row = self.db.query_one(
            "SELECT salt, wrapped_key FROM vault_key_wraps "
            "WHERE credential_type = ?", (credential_type,))
        if not row:
            return None
        try:
            salt = bytes.fromhex(row["salt"])
        except ValueError:
            return None
        kek = self._wrap_kek(credential_type, secret, salt)
        try:
            return aes_decrypt(row["wrapped_key"], key=kek).decode("utf-8")
        except (CipherError, UnicodeDecodeError):
            return None

    def add_key_wrap(self, credential_type: str, secret: str) -> None:
        """Wrap this vault's master key under another owner credential.

        After this call, ``secret`` (of the named type) alone can open
        the vault via :meth:`unlock`."""
        if credential_type not in VAULT_CREDENTIAL_TYPES:
            raise ValueError(
                f"unknown credential type {credential_type!r} — expected "
                f"one of {', '.join(VAULT_CREDENTIAL_TYPES)}")
        if not secret:
            raise ValueError("secret must be non-empty")
        self._wrap_master(credential_type, secret, self._master_key)
        _log.info("vault key wrap added for %s", credential_type)

    def remove_key_wrap(self, credential_type: str) -> None:
        """Drop a credential's wrap (it can no longer open the vault)."""
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM vault_key_wraps WHERE credential_type = ?",
                (credential_type,))

    def _write_verifier(self, master_hex: str) -> None:
        key = derive_key(master_hex, salt=_VERIFIER_SALT,
                         iterations=_KDF_ITERATIONS)
        blob = aes_encrypt(_VERIFIER_PLAINTEXT, key=key)
        with self.db.transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO vault_meta (key, value) "
                "VALUES ('verifier', ?)", (blob,))

    def _master_verified(self, master_hex: str) -> bool:
        """Prove a candidate master key is this vault's real one.

        Uses the verifier blob when present; otherwise authenticates
        against the first stored credential row.  An empty vault with
        no verifier contradicts nothing, so the candidate stands."""
        row = self.db.query_one(
            "SELECT value FROM vault_meta WHERE key = 'verifier'")
        if row:
            key = derive_key(master_hex, salt=_VERIFIER_SALT,
                             iterations=_KDF_ITERATIONS)
            try:
                return aes_decrypt(row["value"], key=key) == _VERIFIER_PLAINTEXT
            except CipherError:
                return False
        row = self.db.query_one(
            "SELECT id, password_encrypted FROM credentials "
            "ORDER BY id LIMIT 1")
        if not row:
            return True
        key = derive_key(master_hex,
                         salt=f"credential-{row['id']}".encode(),
                         iterations=_KDF_ITERATIONS)
        try:
            aes_decrypt(row["password_encrypted"], key=key)
            return True
        except CipherError:
            return False

    def _ensure_wraps(self, available: dict[str, str]) -> list[str]:
        """Migration: wrap the master key under every available owner
        credential that lacks a wrap, and (re)write the verifier.
        Returns the credential types a wrap was added for."""
        added: list[str] = []
        existing = set(self.wrapped_credential_types())
        for ctype in VAULT_CREDENTIAL_TYPES:
            secret = available.get(ctype)
            if secret and ctype not in existing:
                self._wrap_master(ctype, secret, self._master_key)
                added.append(ctype)
        self._write_verifier(self._master_key)
        return added

    @classmethod
    def unlock(cls, db: Database, *, passphrase: str | None = None,
               seal_secret: str | None = None, power_key: str | None = None,
               seal_identity: str = "",
               actor: str = "vault") -> "CredentialVault":
        """Open the vault with ANY one owner credential.

        Candidates are tried in owner order — owner seal, power key,
        passphrase — against the vault's key wraps.  A legacy vault
        (no wraps yet) opens with its passphrase exactly as before and
        is then migrated: wraps are created for every other available
        verified credential.  A brand-new vault with only an owner
        credential available gets a random master key wrapped under
        the credentials at hand.

        Args:
            db: The vault database
            passphrase: Legacy vault passphrase; None falls back to the
                ``NM_VAULT_PASSPHRASE`` environment variable
            seal_secret: The owner seal passphrase, as presented by the
                owner (verified against the baked seal before use —
                an unverified secret is never tried against a wrap)
            power_key: The power-mode key value (``partner.owner_key``)
            seal_identity: Owner identity claim for the seal check
            actor: Audit-log actor name

        Raises:
            VaultLockedError: Every source failed; the message names
                what was checked and where, per source
        """
        if db is None:
            raise VaultLockedError(
                ["database: no database is attached to this context"])
        shell = cls.__new__(cls)
        shell.db = db
        shell._master_key = ""
        shell._ensure_schema()
        checks: list[str] = []

        # ── resolve candidate credentials (values never leave here) ──
        from ..core import owner as _owner

        seal_state = ""
        seal_verified = False
        if not _owner.seal_configured():
            seal_state = ("not configured — no seal is baked in "
                          "(create one with `nm owner seal`)")
        elif not seal_secret:
            seal_state = ("configured, but no seal passphrase was "
                          "presented with this request")
        elif seal_identity and not _owner.verify_owner(seal_identity,
                                                       seal_secret):
            seal_state = ("a seal passphrase was presented but failed "
                          "verification against the baked seal")
        elif not seal_identity and not _owner.verify_passphrase(seal_secret):
            seal_state = ("a seal passphrase was presented but failed "
                          "verification against the baked seal")
        else:
            seal_verified = True

        if passphrase is None:
            passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        available: dict[str, str] = {}
        if seal_verified:
            available["owner_seal"] = seal_secret or ""
        if power_key:
            available["power_key"] = power_key
        if passphrase:
            available["passphrase"] = passphrase

        wraps = shell._wraps()
        has_rows = shell._has_credentials()

        def _finish(master_hex: str, via: str) -> "CredentialVault":
            vault = cls.__new__(cls)
            vault.db = db
            vault._master_key = master_hex
            vault._ensure_schema()
            added = vault._ensure_wraps(available)
            vault._audit_unlock(actor, via, ok=True)
            if added:
                vault._audit_unlock(
                    actor, "vault.migrate", ok=True,
                    why="master key wrapped for: " + ", ".join(added))
                _log.info("vault migration: master key wrapped for %s",
                          ", ".join(added))
            return vault

        # ── route 1: wraps exist — they are authoritative ────────────
        if wraps:
            for ctype in VAULT_CREDENTIAL_TYPES:
                label = _CREDENTIAL_SOURCES[ctype]
                secret = available.get(ctype)
                if secret is None:
                    if ctype == "owner_seal":
                        checks.append(f"{label}: {seal_state}")
                    elif ctype == "power_key":
                        checks.append(
                            f"{label}: not configured — partner.owner_key "
                            "(env NM_PARTNER_OWNER_KEY) is empty")
                    else:
                        checks.append(
                            f"{label}: not set — NM_VAULT_PASSPHRASE is "
                            "absent from the environment and none was supplied")
                    continue
                if ctype not in wraps:
                    checks.append(
                        f"{label}: accepted, but this vault has no "
                        f"{ctype} key wrap yet — one unlock with an "
                        "already-wrapped credential adds it")
                    continue
                master = shell._unwrap_master(ctype, secret)
                if master is not None and shell._master_verified(master):
                    return _finish(master, ctype)
                shell._audit_unlock(actor, ctype, ok=False,
                                    why="key unwrap failed")
                checks.append(
                    f"{label}: the presented credential does not open "
                    "this vault (key unwrap failed)")
            raise VaultLockedError(checks)

        # ── route 2: legacy vault (no wraps) — passphrase derivation ──
        if passphrase:
            vault = cls(db, master_passphrase=passphrase)
            if vault._master_verified(vault._master_key):
                return _finish(vault._master_key, "passphrase")
            shell._audit_unlock(actor, "passphrase", ok=False,
                                why="derived key failed verification")
            checks.append(
                f"{_CREDENTIAL_SOURCES['passphrase']}: a passphrase was "
                "supplied but it does not match this vault (stored "
                "credentials fail authentication under it)")
        elif has_rows:
            checks.append(
                f"{_CREDENTIAL_SOURCES['passphrase']}: not set — "
                "NM_VAULT_PASSPHRASE is absent from the environment and "
                "none was supplied")
            for ctype in ("owner_seal", "power_key"):
                if ctype in available:
                    checks.append(
                        f"{_CREDENTIAL_SOURCES[ctype]}: available, but "
                        "this vault predates multi-credential unlock — "
                        "unlock once with its passphrase and wraps are "
                        "created for the owner credentials automatically")
        else:
            # ── route 3: brand-new vault, owner credential only ──────
            for ctype in ("owner_seal", "power_key"):
                if ctype in available:
                    return _finish(secrets.token_hex(32), ctype)
            checks.append(
                f"{_CREDENTIAL_SOURCES['passphrase']}: not set — "
                "NM_VAULT_PASSPHRASE is absent from the environment and "
                "none was supplied")

        if seal_state and not any("owner seal" in line for line in checks):
            checks.insert(
                0, f"{_CREDENTIAL_SOURCES['owner_seal']}: {seal_state}")
        if "power_key" not in available and not any(
                "power key" in line for line in checks):
            checks.append(
                f"{_CREDENTIAL_SOURCES['power_key']}: not configured — "
                "partner.owner_key (env NM_PARTNER_OWNER_KEY) is empty")
        raise VaultLockedError(checks)

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

    # ── master-passphrase rotation & encrypted backup ──────────────────
    #
    # Bitwarden-grade vaults rotate the master secret by re-encrypting
    # every item. A compromised passphrase without rotation means the
    # vault is dead — so this vault can change its own.

    def change_passphrase(self, new_passphrase: str) -> None:
        """Rotate the vault's master passphrase.

        Every credential is decrypted with the old key and re-encrypted
        under the new one; session blobs (the ``sessions`` table, when
        present) are re-keyed the same way. Atomic: either the whole
        vault moves to the new passphrase or nothing changes.

        Args:
            new_passphrase: The new master passphrase (must be non-empty)

        Raises:
            ValueError: Empty passphrase
            CipherError: A stored row fails authentication mid-rotation
                (the transaction rolls back — nothing is half-rotated)
        """
        if not new_passphrase:
            raise ValueError("new_passphrase must be non-empty")
        old_master = self._master_key
        new_master = derive_key(
            new_passphrase, salt=b"nomorals-vault-master",
            iterations=_KDF_ITERATIONS).hex()
        # Key wraps and the verifier reference the master key: after a
        # rotation the old wraps would wrap a dead key, so below they
        # are dropped and a fresh passphrase wrap is written.  The other
        # owner credentials' wraps are re-created automatically by the
        # next passphrase unlock through CredentialVault.unlock (its
        # migration step), or explicitly via add_key_wrap.
        had_wraps = bool(self.wrapped_credential_types())
        had_verifier = self.db.query_one(
            "SELECT key FROM vault_meta WHERE key = 'verifier'") is not None

        def _old_cred_key(cred_id: int) -> str:
            return derive_key(old_master,
                              salt=f"credential-{cred_id}".encode(),
                              iterations=_KDF_ITERATIONS).hex()

        def _new_cred_key(cred_id: int) -> str:
            return derive_key(new_master,
                              salt=f"credential-{cred_id}".encode(),
                              iterations=_KDF_ITERATIONS).hex()

        from ..core.cipher import CipherError as _CipherError
        with self.db.transaction():
            rows = self.db.query(
                "SELECT id, password_encrypted FROM credentials")
            rotated = []
            for row in rows:
                old_key = bytes.fromhex(_old_cred_key(row["id"]))
                try:
                    from ..core.cipher import aes_decrypt as _dec
                    plain = _dec(row["password_encrypted"],
                                 key=old_key).decode("utf-8")
                except _CipherError as exc:
                    raise _CipherError(
                        f"credential {row['id']} failed authentication "
                        f"during rotation: {exc}") from exc
                new_key = bytes.fromhex(_new_cred_key(row["id"]))
                from ..core.cipher import aes_encrypt as _enc
                rotated.append((_enc(plain.encode("utf-8"), key=new_key),
                                row["id"]))
            for blob, cred_id in rotated:
                self.db.execute(
                    "UPDATE credentials SET password_encrypted = ?, "
                    "updated_at = ? WHERE id = ?",
                    (blob, time.time(), cred_id))
            # Re-key session blobs (purpose "sessions") the same way.
            self._rekey_blobs(old_master, new_master,
                              table="sessions", column="session_data",
                              key_column=("service", "username"),
                              purpose="sessions")
        self._master_key = new_master
        if had_wraps:
            with self.db.transaction():
                self.db.execute("DELETE FROM vault_key_wraps")
            self.add_key_wrap("passphrase", new_passphrase)
        if had_wraps or had_verifier:
            self._write_verifier(new_master)
        _log.info("vault master passphrase rotated (%d credential(s))",
                  len(rotated))

    def _rekey_blobs(self, old_master: str, new_master: str, *,
                     table: str, column: str,
                     key_column: tuple[str, ...], purpose: str) -> int:
        """Re-encrypt every blob in ``table.column`` (purpose-scoped)."""
        from ..core.cipher import (CipherError as _CipherError,
                                   aes_decrypt as _dec, aes_encrypt as _enc)
        try:
            cols = ", ".join(key_column)
            rows = self.db.query(f"SELECT {cols}, {column} FROM {table}")
        except Exception:  # noqa: BLE001 — table may not exist
            return 0
        count = 0
        for row in rows:
            blob = row[column]
            old_key = derive_key(
                old_master,
                salt=f"nomorals-vault-blob:{purpose}".encode(),
                iterations=_KDF_ITERATIONS)
            new_key = derive_key(
                new_master,
                salt=f"nomorals-vault-blob:{purpose}".encode(),
                iterations=_KDF_ITERATIONS)
            try:
                plain = _dec(blob, key=old_key)
            except _CipherError:
                continue  # legacy plaintext row — leave as-is
            new_blob = _enc(plain, key=new_key)
            where = " AND ".join(f"{c} = ?" for c in key_column)
            self.db.execute(
                f"UPDATE {table} SET {column} = ? WHERE {where}",
                (new_blob, *[row[c] for c in key_column]))
            count += 1
        return count

    def export_encrypted(self, path: str, backup_passphrase: str) -> str:
        """Export an encrypted JSON backup of the whole vault.

        All credentials are decrypted in memory and re-encrypted as one
        JSON document under ``backup_passphrase`` (purpose "backup").
        The file is useless without the backup passphrase.

        Args:
            path: Destination file path
            backup_passphrase: Passphrase protecting the backup

        Returns:
            The path written
        """
        if not backup_passphrase:
            raise ValueError("backup_passphrase must be non-empty")
        creds = []
        for summary in self.list_all(active_only=False):
            full = self.get(summary.service, summary.username,
                            mark_used=False)
            d = full.to_dict()
            d["password"] = full.password
            creds.append(d)
        doc = json.dumps({
            "format": "nomorals-vault-backup",
            "version": 1,
            "exported_at": time.time(),
            "credentials": creds,
            "profiles": [p.name for p in self.list_profiles()],
        })
        key = derive_key(
            backup_passphrase,
            salt=b"nomorals-vault-backup",
            iterations=_KDF_ITERATIONS)
        from ..core.cipher import aes_encrypt as _enc
        blob = _enc(doc.encode("utf-8"), key=key)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(blob)
        _log.info("vault backup exported to %s (%d credential(s))",
                  path, len(creds))
        return path

    def import_encrypted(self, path: str, backup_passphrase: str,
                         *, overwrite: bool = False) -> int:
        """Restore credentials from an :meth:`export_encrypted` backup.

        Args:
            path: Backup file path
            backup_passphrase: Passphrase the backup was exported with
            overwrite: Replace existing credentials with backup copies

        Returns:
            Number of credentials imported

        Raises:
            CipherError: Wrong passphrase or tampered backup
            ValueError: Not a vault backup file
        """
        from ..core.cipher import (CipherError as _CipherError,
                                   aes_decrypt as _dec)
        with open(path, encoding="utf-8") as fh:
            blob = fh.read()
        key = derive_key(
            backup_passphrase,
            salt=b"nomorals-vault-backup",
            iterations=_KDF_ITERATIONS)
        try:
            doc = json.loads(_dec(blob, key=key).decode("utf-8"))
        except _CipherError as exc:
            raise _CipherError(
                f"backup decryption failed (wrong passphrase?): {exc}"
            ) from exc
        if doc.get("format") != "nomorals-vault-backup":
            raise ValueError("not a nomorals vault backup file")
        imported = 0
        for c in doc.get("credentials", []):
            exists = self.db.query_one(
                "SELECT id FROM credentials WHERE service = ? AND username = ?",
                (c.get("service", ""), c.get("username", "")))
            if exists and not overwrite:
                continue
            self.store(
                service=c.get("service", ""),
                username=c.get("username", ""),
                password=c.get("password", ""),
                credential_type=c.get("credential_type", "password"),
                tags=list(c.get("tags") or []),
                metadata=dict(c.get("metadata") or {}),
                expires_at=c.get("expires_at"),
            )
            imported += 1
        _log.info("vault backup imported from %s (%d credential(s))",
                  path, imported)
        return imported

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


def open_vault(db: Database, context: Any = None, *,
               seal_secret: str | None = None, seal_identity: str = "",
               actor: str = "vault") -> CredentialVault:
    """Open the accounts vault with whatever owner credential is at hand.

    The shared unlock path for owner-facing surfaces (chat
    provisioning/connectors, the account and connectors CLIs, account
    creation): gathers the power key from ``context.settings.partner``
    (env ``NM_PARTNER_OWNER_KEY``), a presented seal passphrase (the
    argument, or ``context.extras["vault_seal_secret"]`` when a flow
    collected one), and the legacy ``NM_VAULT_PASSPHRASE`` env fallback,
    then delegates to :meth:`CredentialVault.unlock`.  Any ONE accepted
    credential opens the vault — the passphrase is no longer the only
    way in.

    Raises:
        VaultLockedError: No credential could open the vault; the
            message names every source checked and what happened.
    """
    power_key = ""
    settings = getattr(context, "settings", None) if context is not None \
        else None
    partner = getattr(settings, "partner", None) if settings is not None \
        else None
    if partner is not None:
        power_key = str(getattr(partner, "owner_key", "") or "")
    if seal_secret is None and context is not None:
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            presented = extras.get("vault_seal_secret")
            if presented:
                seal_secret = str(presented)
    return CredentialVault.unlock(
        db, seal_secret=seal_secret, seal_identity=seal_identity,
        power_key=power_key or None, actor=actor)
