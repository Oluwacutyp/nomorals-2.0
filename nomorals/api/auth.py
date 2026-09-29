"""API authentication - key management, scopes, rate limiting.

Enhances the basic Bearer token auth with:
- Multiple API keys per user
- Scoped permissions (read, write, admin)
- Key rotation and revocation
- Rate limiting per key
- Usage tracking and audit log
- Key expiration

Usage:
    auth = APIAuth(db)
    
    # Generate a new key
    key = auth.create_key(
        user_id="user_123",
        name="My App",
        scopes=["read", "write"],
        expires_in_days=30,
    )
    print(f"API Key: {key.key}")  # Only shown once!
    
    # Validate a key
    result = auth.validate_key("nm_live_abc123...")
    if result.valid:
        print(f"User: {result.user_id}, Scopes: {result.scopes}")
    
    # Check scope
    if auth.has_scope("nm_live_abc123...", "write"):
        # Allow write operation
        pass
    
    # Revoke a key
    auth.revoke_key("nm_live_abc123...")
    
    # List user's keys
    keys = auth.list_keys("user_123")
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = [
    "APIAuth",
    "APIKey",
    "KeyValidation",
    "KeyScope",
]

_log = get_logger(__name__)


class KeyScope:
    """API key permission scopes."""
    
    READ = "read"          # Read-only access
    WRITE = "write"        # Write access
    ADMIN = "admin"        # Admin access (all permissions)
    CHAT = "chat"          # Chat/conversation access
    AGENTS = "agents"      # Agent execution
    FILES = "files"        # File access
    WEBHOOKS = "webhooks"  # Webhook management
    
    ALL = [READ, WRITE, ADMIN, CHAT, AGENTS, FILES, WEBHOOKS]


@dataclass
class APIKey:
    """An API key with metadata."""
    
    key_id: str
    key_hash: str  # SHA-256 hash of the key
    key_prefix: str  # First 8 chars for identification
    user_id: str
    name: str
    scopes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = never expires
    last_used_at: float = 0.0
    use_count: int = 0
    rate_limit: int = 1000  # Requests per hour
    is_revoked: bool = False
    revoked_at: float = 0.0
    revoke_reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "is_revoked": self.is_revoked,
        }
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        return not self.is_revoked and not self.is_expired


@dataclass
class KeyValidation:
    """Result of validating an API key."""
    
    valid: bool
    key: Optional[APIKey] = None
    error: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "key": self.key.to_dict() if self.key else None,
            "error": self.error,
        }


class APIAuth:
    """API key authentication and authorization."""
    
    KEY_PREFIX_LIVE = "nm_live_"
    KEY_PREFIX_TEST = "nm_test_"
    
    def __init__(self, db: Database, *, test_mode: bool = False) -> None:
        self.db = db
        self.test_mode = test_mode
        self._ensure_schema()
        _log.info(f"APIAuth initialized (test_mode={test_mode})")
    
    def _ensure_schema(self) -> None:
        """Create API auth tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    scopes TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL DEFAULT 0,
                    last_used_at REAL NOT NULL DEFAULT 0,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    rate_limit INTEGER NOT NULL DEFAULT 1000,
                    is_revoked INTEGER NOT NULL DEFAULT 0,
                    revoked_at REAL NOT NULL DEFAULT 0,
                    revoke_reason TEXT NOT NULL DEFAULT ''
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS api_key_usage (
                    usage_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    endpoint TEXT NOT NULL DEFAULT '',
                    method TEXT NOT NULL DEFAULT '',
                    status_code INTEGER NOT NULL DEFAULT 0,
                    ip_address TEXT NOT NULL DEFAULT ''
                )
            """)
            
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_keys_user ON api_keys(user_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_keys_hash ON api_keys(key_hash)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_usage_key ON api_key_usage(key_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_usage_time ON api_key_usage(timestamp)")
    
    def create_key(
        self,
        user_id: str,
        *,
        name: str = "",
        scopes: list[str] | None = None,
        expires_in_days: int = 0,
        rate_limit: int = 1000,
    ) -> APIKey:
        """Create a new API key.
        
        Args:
            user_id: User who owns the key
            name: Human-readable name for the key
            scopes: List of permission scopes
            expires_in_days: Days until expiration (0 = never)
            rate_limit: Requests per hour
            
        Returns:
            APIKey with the plaintext key (only time it's shown!)
        """
        if scopes is None:
            scopes = [KeyScope.READ]
        
        # Validate scopes
        for scope in scopes:
            if scope not in KeyScope.ALL:
                raise ValueError(f"Invalid scope: {scope}")
        
        # Generate key
        prefix = self.KEY_PREFIX_TEST if self.test_mode else self.KEY_PREFIX_LIVE
        plaintext = f"{prefix}{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        key_prefix = plaintext[:12]  # nm_live_x or nm_test_x
        
        key_id = new_id("key")
        created_at = time.time()
        expires_at = created_at + (expires_in_days * 86400) if expires_in_days > 0 else 0.0
        
        # Store key
        import json
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO api_keys
                (key_id, key_hash, key_prefix, user_id, name, scopes,
                 created_at, expires_at, rate_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key_id, key_hash, key_prefix, user_id, name,
                json.dumps(scopes), created_at, expires_at, rate_limit,
            ))
        
        # Create APIKey object with plaintext (only returned once!)
        key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            user_id=user_id,
            name=name,
            scopes=scopes,
            created_at=created_at,
            expires_at=expires_at,
            rate_limit=rate_limit,
        )
        
        # Store plaintext temporarily for return (NOT in DB)
        key._plaintext = plaintext  # type: ignore
        
        _log.info(f"Created API key {key_prefix}... for user {user_id}")
        return key
    
    def validate_key(
        self,
        plaintext_key: str,
        *,
        endpoint: str = "",
        method: str = "",
        ip_address: str = "",
    ) -> KeyValidation:
        """Validate an API key and record usage.
        
        Args:
            plaintext_key: The plaintext API key
            endpoint: API endpoint being accessed
            method: HTTP method
            ip_address: Client IP address
            
        Returns:
            KeyValidation with result
        """
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        
        # Look up key
        row = self.db.query_one("""
            SELECT * FROM api_keys WHERE key_hash = ?
        """, (key_hash,))
        
        if not row:
            return KeyValidation(valid=False, error="Invalid API key")
        
        import json
        key = APIKey(
            key_id=row["key_id"],
            key_hash=row["key_hash"],
            key_prefix=row["key_prefix"],
            user_id=row["user_id"],
            name=row["name"],
            scopes=json.loads(row["scopes"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_used_at=row["last_used_at"],
            use_count=row["use_count"],
            rate_limit=row["rate_limit"],
            is_revoked=bool(row["is_revoked"]),
            revoked_at=row["revoked_at"],
            revoke_reason=row["revoke_reason"],
        )
        
        # Check if valid
        if key.is_revoked:
            return KeyValidation(valid=False, key=key, error=f"Key revoked: {key.revoke_reason}")
        
        if key.is_expired:
            return KeyValidation(valid=False, key=key, error="Key expired")
        
        # Check rate limit
        if not self._check_rate_limit(key):
            return KeyValidation(valid=False, key=key, error="Rate limit exceeded")
        
        # Record usage
        self._record_usage(key, endpoint, method, ip_address)
        
        # Update last used
        with self.db.transaction():
            self.db.execute("""
                UPDATE api_keys
                SET last_used_at = ?, use_count = use_count + 1
                WHERE key_id = ?
            """, (time.time(), key.key_id))
        
        return KeyValidation(valid=True, key=key)
    
    def has_scope(self, plaintext_key: str, required_scope: str) -> bool:
        """Check if a key has a specific scope.
        
        Args:
            plaintext_key: The plaintext API key
            required_scope: Required scope
            
        Returns:
            True if key has the scope
        """
        validation = self.validate_key(plaintext_key)
        
        if not validation.valid or not validation.key:
            return False
        
        # Admin scope grants everything
        if KeyScope.ADMIN in validation.key.scopes:
            return True
        
        return required_scope in validation.key.scopes
    
    def revoke_key(
        self,
        plaintext_key: str,
        *,
        reason: str = "",
    ) -> bool:
        """Revoke an API key.
        
        Args:
            plaintext_key: The plaintext API key
            reason: Reason for revocation
            
        Returns:
            True if key was revoked
        """
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        
        with self.db.transaction():
            cursor = self.db.execute("""
                UPDATE api_keys
                SET is_revoked = 1, revoked_at = ?, revoke_reason = ?
                WHERE key_hash = ? AND is_revoked = 0
            """, (time.time(), reason, key_hash))
            
            if cursor.rowcount > 0:
                _log.info(f"Revoked API key (hash: {key_hash[:8]}...)")
                return True
        
        return False
    
    def revoke_key_by_id(
        self,
        key_id: str,
        *,
        reason: str = "",
    ) -> bool:
        """Revoke an API key by ID."""
        with self.db.transaction():
            cursor = self.db.execute("""
                UPDATE api_keys
                SET is_revoked = 1, revoked_at = ?, revoke_reason = ?
                WHERE key_id = ? AND is_revoked = 0
            """, (time.time(), reason, key_id))
            
            return cursor.rowcount > 0
    
    def list_keys(self, user_id: str) -> list[APIKey]:
        """List all API keys for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of APIKey objects (without plaintext keys)
        """
        import json
        
        rows = self.db.query("""
            SELECT * FROM api_keys
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
        
        keys = []
        for row in rows:
            keys.append(APIKey(
                key_id=row["key_id"],
                key_hash=row["key_hash"],
                key_prefix=row["key_prefix"],
                user_id=row["user_id"],
                name=row["name"],
                scopes=json.loads(row["scopes"]),
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                last_used_at=row["last_used_at"],
                use_count=row["use_count"],
                rate_limit=row["rate_limit"],
                is_revoked=bool(row["is_revoked"]),
                revoked_at=row["revoked_at"],
                revoke_reason=row["revoke_reason"],
            ))
        
        return keys
    
    def get_usage_stats(
        self,
        key_id: str,
        *,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get usage statistics for a key.
        
        Args:
            key_id: API key ID
            days: Number of days to look back
            
        Returns:
            Usage statistics dict
        """
        since = time.time() - (days * 86400)
        
        total = self.db.query_one("""
            SELECT COUNT(*) as count FROM api_key_usage
            WHERE key_id = ? AND timestamp > ?
        """, (key_id, since))
        
        by_status = self.db.query("""
            SELECT status_code, COUNT(*) as count
            FROM api_key_usage
            WHERE key_id = ? AND timestamp > ?
            GROUP BY status_code
        """, (key_id, since))
        
        by_endpoint = self.db.query("""
            SELECT endpoint, COUNT(*) as count
            FROM api_key_usage
            WHERE key_id = ? AND timestamp > ?
            GROUP BY endpoint
            ORDER BY count DESC
            LIMIT 10
        """, (key_id, since))
        
        return {
            "total_requests": total["count"] if total else 0,
            "by_status": {row["status_code"]: row["count"] for row in by_status},
            "top_endpoints": [{"endpoint": row["endpoint"], "count": row["count"]} for row in by_endpoint],
        }
    
    def _check_rate_limit(self, key: APIKey) -> bool:
        """Check if key is within rate limit."""
        one_hour_ago = time.time() - 3600
        
        count = self.db.query_one("""
            SELECT COUNT(*) as count FROM api_key_usage
            WHERE key_id = ? AND timestamp > ?
        """, (key.key_id, one_hour_ago))
        
        return (count["count"] if count else 0) < key.rate_limit
    
    def _record_usage(
        self,
        key: APIKey,
        endpoint: str,
        method: str,
        ip_address: str,
        status_code: int = 200,
    ) -> None:
        """Record API usage."""
        usage_id = new_id("usage")
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO api_key_usage
                (usage_id, key_id, timestamp, endpoint, method, status_code, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (usage_id, key.key_id, time.time(), endpoint, method, status_code, ip_address))
