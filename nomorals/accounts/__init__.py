"""Account management and credential vault.

This module provides secure credential storage, account lifecycle management,
and session handling for the bot's various service integrations.

The bot can:
- Create and manage its own accounts (email, social, etc.)
- Store credentials securely (AES-256 encryption)
- Handle OAuth tokens, API keys, session cookies
- Rotate credentials and manage expiry
- Support multiple identity profiles

Account creation follows the human-in-the-loop checkpoint discipline:
one account per service, the owner's own identity, and human
verification steps (CAPTCHA, email/phone checks) pause on a persisted
checkpoint instead of being bypassed.
"""

from .vault import CredentialVault, Credential, AccountProfile
from .manager import AccountManager
from .creator import (
    AccountCreator,
    CreatedAccount,
    AccountCheckpoint,
    AccountCheckpointPending,
    AccountExistsError,
    MissingOwnerIdentity,
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
)
from .sessions import SessionManager, Session, OAuthToken, SessionInvalid
from .temp_sms import (
    TempSmsProvider,
    SimcodesProvider,
    TempNumber,
    SmsMessage,
    get_provider as get_sms_provider,
    grab_number as grab_temp_number,
    wait_code as wait_temp_sms_code,
    PROVIDERS as SMS_PROVIDERS,
)

__all__ = [
    "CredentialVault",
    "Credential",
    "AccountProfile",
    "AccountManager",
    "AccountCreator",
    "CreatedAccount",
    "AccountCheckpoint",
    "AccountCheckpointPending",
    "AccountExistsError",
    "MissingOwnerIdentity",
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "SessionManager",
    "Session",
    "OAuthToken",
    "SessionInvalid",
    "TempSmsProvider",
    "SimcodesProvider",
    "TempNumber",
    "SmsMessage",
    "get_sms_provider",
    "grab_temp_number",
    "wait_temp_sms_code",
    "SMS_PROVIDERS",
]
