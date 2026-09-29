"""Devon Connector Framework — unified interface for external services.

Every connector exposes the same skeleton:
- status() → live connection state (never cached)
- connect_url() → URL to show user for auth
- disconnect() → revoke access
- refresh() → token refresh where applicable
- capabilities() → ONLY what's actually implemented (scope honesty)

Ten connection patterns supported:
1. OAuth via accounts center
2. Provider-hosted OAuth
3. Platform session link
4. Simple consent flow
5. API key vault (universal fallback)
6. Password vault + browser automation
7. Direct protocols (IMAP/SMTP/CalDAV)
8. Device-local access
9. MCP servers
10. Browser as universal connector

Security rules (non-negotiable):
- Secrets in vault only, never in logs/output/memory/exceptions
- Card PANs/CVVs masked everywhere except checkout handoff
- Webhook signatures verified before acting
- No self-generated card numbers (provider APIs only)
"""

from .base import BaseConnector, ConnectorStatus
from .vault import CredentialVault, VaultError
from .patterns import ConnectionPattern, PatternRegistry

__all__ = [
    "BaseConnector",
    "ConnectorStatus",
    "CredentialVault",
    "VaultError",
    "ConnectionPattern",
    "PatternRegistry",
]
