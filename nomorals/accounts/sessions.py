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

import http.cookiejar
import json
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.cipher import CipherError
from ..core.errors import NoMoralsError, NotFound
from ..core.logging_setup import get_logger
from .vault import Credential, CredentialVault

__all__ = ["SessionManager", "Session", "OAuthToken", "SessionInvalid"]

_log = get_logger(__name__)


class SessionInvalid(NoMoralsError):
    """Raised when a session is missing, expired, or otherwise unusable."""


@dataclass
class OAuthToken:
    """OAuth 2.0 token with metadata."""

    access_token: str = field(repr=False)
    token_type: str = "Bearer"
    expires_at: Optional[float] = None
    refresh_token: Optional[str] = field(default=None, repr=False)
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
    cookies: dict[str, str] = field(default_factory=dict, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    oauth_token: Optional[OAuthToken] = field(default=None, repr=False)
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

    def health_report(self) -> dict[str, Any]:
        """Machine-readable session health.

        Returns ``{"status", "reasons", "age_seconds", "idle_seconds"}``
        where status is one of:

        * ``ok`` — usable right now
        * ``expired`` — OAuth token expired
        * ``stale`` — 24h of inactivity
        * ``empty`` — valid timewise but holds no cookies or OAuth token,
          so it cannot authenticate anything
        """
        reasons: list[str] = []
        now = time.time()
        if self.oauth_token is not None and self.oauth_token.is_expired():
            reasons.append("oauth_token_expired")
        if now - self.last_used > 24 * 3600:
            reasons.append("idle_over_24h")
        if not self.cookies and (
            self.oauth_token is None or self.oauth_token.is_expired()
        ):
            reasons.append("no_cookies_or_token")
        if reasons:
            status = "expired" if "oauth_token_expired" in reasons else (
                "stale" if "idle_over_24h" in reasons else "empty")
        else:
            status = "ok"
        return {
            "status": status,
            "reasons": reasons,
            "age_seconds": now - self.created_at,
            "idle_seconds": now - self.last_used,
        }
    
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
    
    def _load_session(
        self, service: str, username: str
    ) -> Session | None:
        """Load a session from cache/database WITHOUT touching it.

        The returned session keeps its stored ``last_used``, so callers
        can judge staleness honestly.
        """
        key = self._session_key(service, username)
        if key in self._sessions:
            return self._sessions[key]
        row = self.vault.db.query_one(
            "SELECT * FROM sessions WHERE service = ? AND username = ?",
            (service, username)
        )
        if row:
            session = Session.from_dict(
                json.loads(self._decrypt_session_data(row["session_data"]))
            )
            self._sessions[key] = session
            return session
        return None

    def peek_session(self, service: str, username: str) -> Session | None:
        """Load the stored session WITHOUT touching it.

        Health checks and audits use this so a stale session isn't
        accidentally refreshed by the act of looking at it.
        """
        return self._load_session(service, username)

    def get_session(self, service: str, username: str) -> Session:
        """Get or create a session for a service.

        Lenient by design: a stored-but-stale session is refreshed via
        touch (use :meth:`get_valid_session` for the strict variant).

        Args:
            service: Service name
            username: Username or identifier

        Returns:
            Session object
        """
        key = self._session_key(service, username)
        session = self._load_session(service, username)
        if session is None:
            session = Session(service=service, username=username)
            self._sessions[key] = session
        session.touch()
        self._save_session(session)
        return session

    def ensure_authenticated(
        self,
        service: str,
        username: str,
        *,
        reauth: Optional[Callable[[], Any]] = None,
    ) -> Session:
        """Return a usable session, re-authenticating when it is gone.

        Unlike :meth:`get_valid_session` (which just raises), this tries
        the ``reauth`` callable once when the stored session is
        invalid/expired: ``reauth`` performs whatever flow restores the
        session (typically :func:`browser_login.ensure_login` with a
        zero-arg tab factory bound) and may return anything — the
        session is re-loaded from the store afterwards.

        Args:
            service: Service name
            username: Username or identifier
            reauth: Optional zero-arg callable that restores the session

        Returns:
            A valid Session

        Raises:
            SessionInvalid: Session unusable and re-auth missing/failed
        """
        try:
            return self.get_valid_session(service, username)
        except SessionInvalid as exc:
            if reauth is None:
                raise
            _log.info("session for %s/%s invalid (%s) — running re-auth",
                      service, username, exc)
        try:
            reauth()
        except Exception as exc:  # noqa: BLE001 — report, then re-raise below
            _log.warning("re-auth for %s/%s failed: %s", service, username,
                         exc)
        # One more attempt: either the re-auth restored the session, or
        # this raises SessionInvalid with the real reason.
        return self.get_valid_session(service, username)

    def probe(
        self,
        service: str,
        username: str,
        url: str,
        *,
        ok_markers: tuple[str, ...] | list[str] = (),
        bad_markers: tuple[str, ...] | list[str] = (),
        marker_status: dict[str, tuple[str, ...] | list[str]] | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Server-side session check: fetch ``url`` with the session's
        cookies and look for login-state markers.

        A session can look valid locally (fresh, cookies present) while
        the server has already killed it — this is how that is detected.

        Args:
            service: Service name
            username: Username or identifier
            url: A page/API endpoint that only a logged-in user can see
            ok_markers: Text that must appear when logged in
            bad_markers: Text that means logged out (e.g. "sign in",
                "log in to continue")
            marker_status: Optional mapping status -> markers, checked
                BEFORE ``bad_markers`` so a page can be classified more
                precisely than "logged out" (e.g. {"locked": (...),
                "needs_verification": (...)})
            timeout: HTTP timeout in seconds

        Returns:
            ``{"ok": True/False, "status": ..., "http_status": int|None,
            "reason": str}`` — status is "logged_in" when the server
            accepts the session, "logged_out" on rejection, a
            ``marker_status`` key on a classified page, or "unknown" on
            transport/HTTP errors. Never raises for transport errors.
        """
        session = self.get_session(service, username)
        jar = http.cookiejar.CookieJar()
        parsed = urllib.parse.urlparse(url)
        for name, value in (session.cookies or {}).items():
            jar.set_cookie(http.cookiejar.Cookie(
                version=0, name=name, value=value,
                port=None, port_specified=False,
                domain=parsed.hostname or "", domain_specified=bool(parsed.hostname),
                domain_initial_dot=False, path="/", path_specified=True,
                secure=False, expires=None, discard=True,
                comment=None, comment_url=None, rest={},
            ))
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar))
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with opener.open(req, timeout=timeout) as resp:
                http_status: int | None = resp.status
                body = resp.read(200_000).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            try:
                body = exc.read(200_000).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 — best-effort body
                body = ""
        except Exception as exc:  # noqa: BLE001 — transport failure
            return {"ok": False, "status": "unknown", "http_status": None,
                    "reason": f"transport error: {exc}"}
        low = body.lower()
        for status, markers in (marker_status or {}).items():
            for marker in markers:
                if marker and marker.lower() in low:
                    return {"ok": False, "status": status,
                            "http_status": http_status,
                            "reason": f"page marker {marker!r} "
                                      f"(classified {status})"}
        for marker in bad_markers:
            if marker and marker.lower() in low:
                return {"ok": False, "status": "logged_out",
                        "http_status": http_status,
                        "reason": f"logout marker {marker!r} on page"}
        for marker in ok_markers:
            if marker and marker.lower() not in low:
                return {"ok": False, "status": "unknown",
                        "http_status": http_status,
                        "reason": f"expected marker {marker!r} missing"}
        if http_status in (401, 403):
            return {"ok": False, "status": "logged_out",
                    "http_status": http_status,
                    "reason": f"HTTP {http_status} from server"}
        if http_status and http_status >= 400:
            return {"ok": False, "status": "unknown",
                    "http_status": http_status,
                    "reason": f"HTTP {http_status} from server"}
        return {"ok": True, "status": "logged_in", "http_status": http_status,
                "reason": "session accepted by server"}
    
    def _encrypt_session_data(self, plaintext: str) -> str:
        """Encrypt serialized session data (may hold OAuth tokens)."""
        return self.vault.encrypt_blob(plaintext, purpose="sessions")

    def _decrypt_session_data(self, blob: str) -> str:
        """Decrypt session data, accepting legacy plaintext rows.

        Rows written before encryption existed are read as-is and
        re-saved encrypted on the next touch.
        """
        try:
            return self.vault.decrypt_blob(blob, purpose="sessions")
        except CipherError:
            return blob  # legacy plaintext row

    def _save_session(self, session: Session) -> None:
        """Persist session to database (session_data encrypted at rest)."""
        with self.vault.db.transaction():
            self.vault.db.execute("""
                INSERT OR REPLACE INTO sessions (service, username, session_data, created_at, last_used)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.service,
                session.username,
                self._encrypt_session_data(json.dumps(session.to_dict())),
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
    
    def get_valid_session(self, service: str, username: str) -> Session:
        """Get the session, failing fast when it is expired/invalid.

        Unlike :meth:`get_session` (which happily returns a stale
        session), this raises :class:`SessionInvalid` when the session
        would not authenticate — expired OAuth token or 24h of
        inactivity.

        Raises:
            SessionInvalid: Session is not usable
        """
        session = self._load_session(service, username)
        if session is None:
            # Nothing stored: create a fresh, valid session.
            return self.get_session(service, username)
        if not session.is_valid():
            if (session.oauth_token is not None
                    and session.oauth_token.is_expired()):
                raise SessionInvalid(
                    f"OAuth token expired for {service}/{username} — "
                    "refresh it (ensure_oauth_token) or re-authenticate"
                )
            raise SessionInvalid(
                f"Session expired for {service}/{username} "
                "(24h inactivity) — re-authenticate"
            )
        session.touch()
        self._save_session(session)
        return session

    def auth_headers(self, service: str, username: str) -> dict[str, str]:
        """Build request headers for an authenticated call.

        Merges the session's stored headers with an ``Authorization``
        bearer header when a valid OAuth token is present.

        Args:
            service: Service name
            username: Username or identifier

        Returns:
            Dict of header name -> value
        """
        session = self.get_session(service, username)
        headers = dict(session.headers)
        token = session.oauth_token
        if token is not None and not token.is_expired():
            headers.setdefault(
                "Authorization", f"{token.token_type} {token.access_token}"
            )
        return headers

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
        *,
        auto_refresh: dict[str, str] | None = None,
    ) -> None:
        """Set OAuth token for a session.

        Args:
            service: Service name
            username: Username or identifier
            token: OAuthToken object
            auto_refresh: Optional refresh configuration
                (``token_url``, ``client_id``, ``client_secret``,
                ``refresh_token``). Stored in session metadata so
                :meth:`ensure_oauth_token` can refresh automatically.
                Omit the client secret when the flow doesn't need it;
                the session row is encrypted at rest.
        """
        session = self.get_session(service, username)
        session.oauth_token = token
        if auto_refresh is not None:
            session.metadata["oauth_auto_refresh"] = dict(auto_refresh)
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
    
    def ensure_oauth_token(
        self,
        service: str,
        username: str,
    ) -> OAuthToken | None:
        """Return a usable OAuth token, refreshing it if needed.

        * No token stored → returns None.
        * Token valid → returned as-is.
        * Token expired and an ``auto_refresh`` config was stored with
          :meth:`set_oauth_token` → refreshed via the token endpoint
          and the new token returned.
        * Token expired without refresh config → raises
          :class:`SessionInvalid`.

        Raises:
            SessionInvalid: Token expired and cannot be refreshed
        """
        session = self.get_session(service, username)
        token = session.oauth_token
        if token is None:
            return None
        if not token.is_expired():
            return token
        cfg = session.metadata.get("oauth_auto_refresh") or {}
        missing = [k for k in ("token_url", "client_id")
                   if not cfg.get(k)]
        refresh_token = token.refresh_token or cfg.get("refresh_token", "")
        if missing or not refresh_token:
            raise SessionInvalid(
                f"OAuth token expired for {service}/{username} and no "
                f"usable refresh configuration is stored "
                f"(missing: {', '.join(missing) or 'refresh_token'})"
            )
        return self.refresh_oauth_token(
            service,
            username,
            refresh_token,
            cfg["client_id"],
            cfg.get("client_secret", ""),
            cfg["token_url"],
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

        # Drop the encrypted OAuth copy kept in the vault, if any.
        try:
            self.vault.delete(f"{service}_oauth", username)
        except NotFound:
            _log.debug("no vault OAuth copy for %s/%s", service, username)

        _log.info(f"Cleared session: {service}/{username}")
    
    def list_sessions(self, *, valid_only: bool = False) -> list[Session]:
        """List stored sessions.

        Args:
            valid_only: If True, only sessions that would still
                authenticate (unexpired token, recent activity)

        Returns:
            List of Session objects, most-recently-used first
        """
        rows = self.vault.db.query("SELECT * FROM sessions ORDER BY last_used DESC")
        sessions = []

        for row in rows:
            session = Session.from_dict(
                json.loads(self._decrypt_session_data(row["session_data"]))
            )
            if valid_only and not session.is_valid():
                continue
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
