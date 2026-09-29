"""Ten connection patterns — every connector picks ≥2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["ConnectionPattern", "PatternRegistry"]


class ConnectionPattern(str, Enum):
    """Supported connection patterns."""
    
    OAUTH_ACCOUNTS_CENTER = "oauth_accounts_center"
    OAUTH_PROVIDER_HOSTED = "oauth_provider_hosted"
    SESSION_LINK = "session_link"
    CONSENT_FLOW = "consent_flow"
    API_KEY_VAULT = "api_key_vault"
    PASSWORD_VAULT_BROWSER = "password_vault_browser"
    DIRECT_PROTOCOL = "direct_protocol"
    DEVICE_LOCAL = "device_local"
    MCP_SERVER = "mcp_server"
    BROWSER_FALLBACK = "browser_fallback"


@dataclass
class PatternConfig:
    """Configuration for a connection pattern."""
    
    pattern: ConnectionPattern
    description: str
    setup_fn: Any = None  # Callable to initiate connection
    refresh_fn: Any = None  # Callable to refresh tokens
    revoke_fn: Any = None  # Callable to revoke access


class PatternRegistry:
    """Registry of connection pattern implementations."""
    
    def __init__(self) -> None:
        self._patterns: dict[ConnectionPattern, PatternConfig] = {}
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default pattern implementations."""
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.OAUTH_ACCOUNTS_CENTER,
            description="OAuth via accounts center — user signs in at provider, grant managed centrally",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.OAUTH_PROVIDER_HOSTED,
            description="Provider-hosted OAuth — authorize_url → exchange_code → refresh",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.SESSION_LINK,
            description="Platform session link — piggyback existing session (cookies/token bridge)",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.CONSENT_FLOW,
            description="Simple consent flow — no sign-in, just permission grant (read-only public data)",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.API_KEY_VAULT,
            description="API key capture — hosted secure page, user pastes key, stored encrypted (universal fallback)",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.PASSWORD_VAULT_BROWSER,
            description="Password vault + browser automation — saved login, bot drives real website",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.DIRECT_PROTOCOL,
            description="Direct protocols — IMAP/SMTP/CalDAV: host + port + app password",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.DEVICE_LOCAL,
            description="Device-local access — paired phone APIs, desktop app bridges (no cloud auth)",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.MCP_SERVER,
            description="MCP servers — OAuth-backed MCP, Devon can expose and consume these",
        ))
        
        self.register(PatternConfig(
            pattern=ConnectionPattern.BROWSER_FALLBACK,
            description="Browser as universal connector — signed-in browser session (slowest but covers everything)",
        ))
    
    def register(self, config: PatternConfig) -> None:
        """Register a pattern implementation."""
        self._patterns[config.pattern] = config
    
    def get(self, pattern: ConnectionPattern) -> PatternConfig | None:
        """Get pattern config."""
        return self._patterns.get(pattern)
    
    def list(self) -> list[PatternConfig]:
        """List all registered patterns."""
        return list(self._patterns.values())
    
    def describe(self, pattern: ConnectionPattern) -> str:
        """Get pattern description."""
        config = self.get(pattern)
        return config.description if config else "Unknown pattern"


# Module-level registry
patterns = PatternRegistry()
