"""Account management and credential vault.

This module provides secure credential storage, account lifecycle management,
and session handling for the bot's various service integrations.

The bot can:
- Create and manage its own accounts (email, social, etc.)
- Store credentials securely (AES-256 encryption)
- Handle OAuth tokens, API keys, session cookies
- Rotate credentials and manage expiry
- Support multiple identity profiles
"""

from .vault import CredentialVault, Credential, AccountProfile
from .manager import AccountManager
from .creator import AccountCreator
from .sessions import SessionManager

__all__ = [
    "CredentialVault",
    "Credential",
    "AccountProfile",
    "AccountManager",
    "AccountCreator",
    "SessionManager",
]
