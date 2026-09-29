"""Base connector interface — every connector implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ConnectorStatus:
    """Live connection state — always fresh, never cached."""
    
    connected: bool
    account: str | None = None
    scopes: list[str] = field(default_factory=list)
    last_check: str = ""
    error: str = ""
    
    def __post_init__(self) -> None:
        if not self.last_check:
            self.last_check = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "account": self.account,
            "scopes": self.scopes,
            "last_check": self.last_check,
            "error": self.error,
        }


class BaseConnector(ABC):
    """Abstract base for all connectors.
    
    Every connector must implement:
    - status() → ConnectorStatus (live check, never cached)
    - connect_url() → str (URL to show user, or "" if none)
    - disconnect() → dict (revocation result)
    
    Optional overrides:
    - refresh() → dict (token refresh, default: no-op)
    - capabilities() → list[str] (ONLY what's implemented)
    
    Hard rules:
    - status() MUST check live every call
    - capabilities() MUST only list implemented features
    - Never invent connect URLs
    - Secrets stay in vault, never in logs/output/exceptions
    """
    
    name: str = ""
    description: str = ""
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        self.vault = vault
        self.config = config or {}
    
    @abstractmethod
    def status(self) -> ConnectorStatus:
        """Check live connection state. NEVER cache this."""
        ...
    
    @abstractmethod
    def connect_url(self) -> str:
        """URL to show user for connection. Return "" if none applies."""
        ...
    
    @abstractmethod
    def disconnect(self) -> dict[str, Any]:
        """Revoke access and clean up."""
        ...
    
    def refresh(self) -> dict[str, Any]:
        """Refresh tokens where applicable. Default: no-op."""
        return {"refreshed": False, "reason": "n/a"}
    
    def capabilities(self) -> list[str]:
        """List ONLY implemented capabilities. Scope honesty is mandatory."""
        return []
    
    def _get_credential(self, label: str) -> dict[str, Any] | None:
        """Retrieve credential from vault. Returns None if not found."""
        if not self.vault or not self.name:
            return None
        try:
            return self.vault.get(self.name, label)
        except Exception:
            return None
    
    def _store_credential(self, label: str, secret_dict: dict[str, Any]) -> bool:
        """Store credential in vault. Returns True on success."""
        if not self.vault or not self.name:
            return False
        try:
            self.vault.store(self.name, label, secret_dict)
            return True
        except Exception:
            return False
    
    def _delete_credential(self, label: str) -> bool:
        """Delete credential from vault. Returns True on success."""
        if not self.vault or not self.name:
            return False
        try:
            self.vault.delete(self.name, label)
            return True
        except Exception:
            return False
