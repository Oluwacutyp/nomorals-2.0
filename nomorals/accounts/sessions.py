"""Session management for authenticated services.

Handles:
- Cookie persistence across requests
- OAuth token refresh flows
- Session state management
- Login/logout flows

Usage:
    sessions = SessionManager(vault)
    
    # Get or create a session for a service
    session = sessions.get_session("github", "my-bot")
    
    # Use the session for authenticated requests
    response = session.get("https://api.github.com/user")
    
    # Refresh expired tokens
    sessions.refresh_token("gmail", "bot@example.com")
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.logging_setup import get_logger
from .vault import Credential, CredentialVault

__all__ = ["SessionManager", "Session", "OAuthToken"]

_log = get_logger(__name__)


@dataclass
class OAuthToken:
    """OAuth 2.0 token with metadata."""
    
    access_token: str
    token_type: str = "Bearer"
    expires_at: Optional[float] = None
    refresh_token: Optional[str] = None
    scope: str = ""
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthToken":
        """Deserialize from dict."""
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=data.get("expires_at"),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope", ""),
        )


@dataclass
class Session:
    """An authenticated session for a service."""
    
    service: str
    username: str
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    oauth_token: Optional[OAuthToken] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_used: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_used == 0.0:
            self.last_used = time.time()
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        if self.oauth_token and self.oauth_token.is_expired():
            return False
        # Session expires after 24 hours of inactivity
        if time.time() - self.last_used > 24 * 3600:
            return False
        return True
    
    def touch(self):
        """Update last_used timestamp."""
        self.last_used = time.time()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "service": self.service,
            "username": self.username,
            "cookies": self.cookies,
            "headers": self.headers,
            "oauth_token": self.oauth_token.to_dict() if self.oauth_token else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Deserialize from dict."""
        oauth_data = data.get("oauth_token")
        oauth_token = OAuthToken.from_dict(oauth_data) if oauth_data else None
        
        return cls(
            service=data["service"],
            username=data["username"],
            cookies=data.get("cookies", {}),
            headers=data.get("headers", {}),
            oauth_token=oauth_token,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", time.time()),
        )


class SessionManager:
    """Manages authenticated sessions for various services.
    
    Sessions are cached in memory and persisted to vault.
    """
    
    def __init__(self, vault: CredentialVault) -> None:
        self.vault = vault
        self._sessions: dict[str, Session] = {}  # key: "service/username"
        self._ensure_schema()
        _log.info("Session manager initialized")
    
    def _ensure_schema(self) -> None:
        """Create sessions table if it doesn't exist."""
        with self.vault.db.transaction():
            self.vault.db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    service TEXT NOT NULL,
                    username TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL,
                    PRIMARY KEY (service, username)
                )
            """)
    
    def _session_key(self, service: str, username: str) -> str:
        """Generate session cache key."""
        return f"{service}/{username}"
    
    def get_session(self, service: str, username: str) -> Session:
        """Get or create a session for a service.
        
        Args:
            service: Service name
            username: Username or identifier
            
        Returns:
            Session object
        """
        key = self._session_key(service, username)
        
        # Check cache first
        if key in self._sessions:
            session = self._sessions[key]
            session.touch()
            return session
        
        # Try to load from database
        row = self.vault.db.query_one(
            "SELECT * FROM sessions WHERE service = ? AND username = ?",
            (service, username)
        )
        
        if row:
            session = Session.from_dict(json.loads(row["session_data"]))
            self._sessions[key] = session
            session.touch()
            return session
        
        # Create new session
        session = Session(service=service, username=username)
        self._sessions[key] = session
        self._save_session(session)
        
        return session
    
    def _save_session(self, session: Session) -> None:
        """Persist session to database."""
        with self.vault.db.transaction():
            self.vault.db.execute("""
                INSERT OR REPLACE INTO sessions (service, username, session_data, created_at, last_used)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.service,
                session.username,
                json.dumps(session.to_dict()),
                session.created_at,
                session.last_used,
            ))
    
    def update_session(self, session: Session) -> None:
        """Update an existing session.
        
        Args:
            session: Session object to update
        """
        session.touch()
        key = self._session_key(session.service, session.username)
        self._sessions[key] = session
        self._save_session(session)
    
    def set_cookies(self, service: str, username: str, cookies: dict[str, str]) -> None:
        """Set cookies for a session.
        
        Args:
            service: Service name
            username: Username or identifier
            cookies: Dict of cookie name -> value
        """
        session = self.get_session(service, username)
        session.cookies.update(cookies)
        self.update_session(session)
    
    def set_headers(self, service: str, username: str, headers: dict[str, str]) -> None:
        """Set headers for a session.
        
        Args:
            service: Service name
            username: Username or identifier
            headers: Dict of header name -> value
        """
        session = self.get_session(service, username)
        session.headers.update(headers)
        self.update_session(session)
    
    def set_oauth_token(
        self,
        service: str,
        username: str,
        token: OAuthToken,
    ) -> None:
        """Set OAuth token for a session.
        
        Args:
            service: Service name
            username: Username or identifier
            token: OAuthToken object
        """
        session = self.get_session(service, username)
        session.oauth_token = token
        self.update_session(session)
        
        # Also store in vault for persistence
        self.vault.store(
            service=f"{service}_oauth",
            username=username,
            password=json.dumps(token.to_dict()),
            credential_type="oauth_token",
            tags=[service, "oauth"],
            expires_at=token.expires_at,
        )
    
    def refresh_oauth_token(
        self,
        service: str,
        username: str,
        refresh_token: str,
        client_id: str,
        client_secret: str,
        token_url: str,
    ) -> OAuthToken:
        """Refresh an expired OAuth token.
        
        Args:
            service: Service name
            username: Username or identifier
            refresh_token: Refresh token
            client_id: OAuth client ID
            client_secret: OAuth client secret
            token_url: Token endpoint URL
            
        Returns:
            New OAuthToken object
        """
        # Prepare refresh request
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }).encode()
        
        req = urllib.request.Request(
            token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                
                # Parse token response
                expires_in = result.get("expires_in", 3600)
                new_token = OAuthToken(
                    access_token=result["access_token"],
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=time.time() + expires_in,
                    refresh_token=result.get("refresh_token", refresh_token),
                    scope=result.get("scope", ""),
                )
                
                # Update session
                self.set_oauth_token(service, username, new_token)
                _log.info(f"Refreshed OAuth token for {service}/{username}")
                
                return new_token
        except Exception as e:
            _log.error(f"Failed to refresh OAuth token: {e}")
            raise
    
    def clear_session(self, service: str, username: str) -> None:
        """Clear a session (logout).
        
        Args:
            service: Service name
            username: Username or identifier
        """
        key = self._session_key(service, username)
        
        # Remove from cache
        if key in self._sessions:
            del self._sessions[key]
        
        # Remove from database
        with self.vault.db.transaction():
            self.vault.db.execute(
                "DELETE FROM sessions WHERE service = ? AND username = ?",
                (service, username)
            )
        
        _log.info(f"Cleared session: {service}/{username}")
    
    def list_sessions(self) -> list[Session]:
        """List all active sessions.
        
        Returns:
            List of Session objects
        """
        rows = self.vault.db.query("SELECT * FROM sessions ORDER BY last_used DESC")
        sessions = []
        
        for row in rows:
            session = Session.from_dict(json.loads(row["session_data"]))
            sessions.append(session)
        
        return sessions
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        sessions = self.list_sessions()
        removed = 0
        
        for session in sessions:
            if not session.is_valid():
                self.clear_session(session.service, session.username)
                removed += 1
        
        if removed > 0:
            _log.info(f"Cleaned up {removed} expired sessions")
        
        return removed
